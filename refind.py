"""refind: fingerprint an element of a browser-use 1.0 page listing, find it again later.

browser-use 1.0 identifies elements by Chrome backendNodeId. Those ids are only valid for one
page load, so a recorded `bu.click(1400)` cannot be replayed as written. `fingerprint` describes
the element by what the listing says about it (role, name, heading, and the rare words on the
lines around it). `find` looks for it in a fresh listing and returns the new id, or refuses.

CLI:
    python refind.py compile transcript.json          a replay plan as JSON
    python refind.py find fingerprint.json state.txt  the new id, or the reason for refusing
"""
import json
import re
import sys
import zlib
from difflib import SequenceMatcher

from axstate import fold, parse_state

VERSION = 2
BEFORE, AFTER = 5, 3          # context lines kept on each side
DECAY = 0.6                   # weight of a context line falls off with its distance
SHIFT = 0.8                   # and a matching line that moved costs this much per position
SIDE = 0.75                   # share of the context score given to the lines before the element
ROW_CAP = 6                   # most lines of one row kept on each side
EDGE = "<edge>"               # pseudo token for the start or end of the document
NUMERIC = 0.3                 # numbers (prices, counts, "3 days ago") change often, so they weigh less
EXTRA = 0.5                   # cost of each unrecorded token on a matching line ("Sun Hat" vs "Sun Hat 2")
# Roles a redesign swaps without changing what the control does.
ROLE_FAMILIES = [{"button", "link", "menuitem", "tab"}, {"textbox", "searchbox", "combobox"},
                 {"checkbox", "switch"}, {"select", "listbox", "combobox"}]
TOGGLES = {"checkbox", "switch", "radio"}

# Weights and thresholds. Chosen by `python bench.py --tune --quick` (random search, 400 samples) on
# the TRAIN generator only: the shop and admin sites, train drift mix, seeds 0-7. The objective
# counts a wrong match as 25 refusals. The search picked solo_ctx 0.0 from its grid; I then tried
# 0.05 on the same train cases because two wrong matches had no context at all, and kept it. The
# heading condition next to solo_ctx in `decide` was also added by hand after reading train failures.
# bench.py reports these values on the held-out sites as well.
PARAMS = {"w_name": 2, "w_ctx": 5, "w_head": 0.5, "w_ord": 0.25, "threshold": 0.7, "margin": 0.05,
          "ctx_min": 0.5, "rename_ctx": 0.6, "solo_ctx": 0.05, "rename_name": 0.2}


def tokens(s):
    return re.findall(r"[a-z0-9]+(?:[.@_-][a-z0-9]+)*", fold(s))


def _line_tokens(state, i):
    kind, text, entry = state.lines[i]
    if kind == "control":
        return tokens(f"{entry.role} {entry.name}")
    return tokens(text) if kind in ("text", "heading") else []


def _window(state, at, step, limit):
    """Token lists of up to `limit` lines next to line `at`, nearest first. The edge of the
    document counts as a line of its own, so "last thing on the page" is evidence too."""
    out = []
    i = at + step
    while len(out) < limit:
        if not 0 <= i < len(state.lines) or state.lines[i][0] == "frame":
            out.append([EDGE])
            break
        toks = _line_tokens(state, i)
        if toks:
            out.append(toks)
        i += step
    return out


def _twins(state, entry):
    return [e for e in state.entries
            if e.role == entry.role and fold(e.name) == fold(entry.name) and e.frame == entry.frame]


def _name_counts(state):
    counts = {}
    for e in state.entries:
        key = (e.role, fold(e.name), e.frame)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _row(state, entry, twins, counts, label_after=False, cap=None):
    """The lines that belong with `entry` when its role and name repeat down the page (Edit, Edit, Edit).

    Returns (label side, other side, non-generic lines on the label side); the sides are token
    lists, nearest line first.

    A line is "generic" when it is a control whose role and name repeat too (the Remove next to
    every Edit). Walking from the element towards the neighbouring twin, the row reaches as far
    as the farthest line that is not generic: the row's own text or its uniquely named link.
    Generic controls beyond that belong to the neighbouring row. On the other side the row
    continues while the lines are generic controls.

    No row takes more non-generic lines than `cap`: by default the usual number in this list, so
    a toolbar above a table is not part of row one; when matching, the number the recorded row had.

    label_after=False walks backwards for the label, which fits a row whose text comes first.
    `_label_after` decides when to walk forwards instead.
    """
    at = [t.line for t in twins]
    k = at.index(entry.line)
    step = 1 if label_after else -1

    def generic(i):
        kind, _, e = state.lines[i]
        return kind == "control" and counts[(e.role, fold(e.name), e.frame)] > 1

    def walk(line, stop, step):
        out = []
        for i in range(line + step, stop, step):
            if state.lines[i][0] in ("heading", "frame"):
                break
            out.append(i)
        return out

    def own(idx):
        far = max((j for j, i in enumerate(idx) if not generic(i)), default=-1)
        return idx[:far + 1]

    edge = len(state.lines) if label_after else -1
    if cap is None:  # the usual number of non-generic lines in a row of this list
        sizes = sorted(sum(not generic(i) for i in own(walk(at[m], at[m + step], step)))
                       for m in range(len(at)) if 0 <= m + step < len(at))
        cap = sizes[len(sizes) // 2]
    j = k + step
    n, label = 0, []
    # Stop at the neighbouring twin, and also after `cap` non-generic lines: a neighbouring row
    # that has lost its own twin (an item with no "Add to cart" any more) must not be read as
    # part of this row.
    for i in walk(entry.line, at[j] if 0 <= j < len(at) else edge, step):
        if not generic(i):
            if n == cap:
                break
            n += 1
        label.append(i)
    label = own(label)
    j = k - step
    stop = at[j] if 0 <= j < len(at) else (-1 if label_after else len(state.lines))
    other = []
    for i in walk(entry.line, stop, -step):
        if not generic(i):
            break
        other.append(i)
    return ([_line_tokens(state, i) for i in label[:ROW_CAP]], [_line_tokens(state, i) for i in other[:ROW_CAP]],
            sum(not generic(i) for i in label))


def _label_after(state, twins, counts):
    """True when the rows keep their label after their controls.

    The listing gives one hint: if nothing but the heading stands in front of the first twin,
    while the last twin is followed by text, the text cannot be in front.
    """
    if _row(state, twins[0], twins, counts)[0]:
        return False
    return bool(_row(state, twins[-1], twins, counts, label_after=True)[0])


def _doc_freq(state):
    df = {}
    for i in range(len(state.lines)):
        for t in set(_line_tokens(state, i)):
            df[t] = df.get(t, 0) + 1
    return df


def fingerprint(state, id):
    """A JSON-serialisable description of the control `id` in `state`. No id is stored."""
    e = state.by_id(id)
    if e is None:
        raise KeyError(f"id {id} is not in this state")
    twins = _twins(state, e)
    label_after, row_size = False, 0
    if len(twins) > 1:
        # The role and name repeat, so only the row says which one this is. A token counts by how
        # few of the other rows also have it: the row's own label counts 1, "button" counts 0.
        counts = _name_counts(state)
        label_after = _label_after(state, twins, counts)
        rows = [_row(state, t, twins, counts, label_after) for t in twins]
        mine = rows[twins.index(e)]
        row_size = mine[2]

        def weigh(side):
            have = [set(t for line in r[side] for t in line) for r in rows]
            share = lambda t: (sum(t in h for h in have) - 1) / (len(twins) - 1)
            return [{t: round((1.0 - share(t)) * _volatile(t), 4) for t in sorted(set(line))} for line in mine[side]]

        before, after = weigh(0), weigh(1)
    else:
        # Unique on the page: weigh nearby tokens by how rare they are on the whole page.
        df = _doc_freq(state)
        weigh = lambda side: [{t: round(_volatile(t) / df.get(t, 1), 4) for t in sorted(set(line))} for line in side]
        before, after = weigh(_window(state, e.line, -1, BEFORE)), weigh(_window(state, e.line, +1, AFTER))
    return {
        "v": VERSION, "role": e.role, "name": e.name, "name_clipped": e.name_clipped,
        "heading": e.heading, "frame": e.frame, "hidden": e.hidden,
        "states": [s for s in e.states if s in ("checked", "unchecked", "expanded", "collapsed", "selected")],
        "twins": len(twins), "ordinal": twins.index(e),
        # For a repeated control: the row's label side and its other side (see _row). For a unique
        # one: the lines before and after it.
        "before": before, "after": after, "label_after": label_after, "row_size": row_size,
        # Every role and name on the recorded page, hashed. A differently named candidate that was
        # already on the page as something else cannot be this element after a relabel.
        "known": sorted({_name_key(x.role, x.name) for x in state.entries}),
    }


def _volatile(token):
    return NUMERIC if token.isdigit() else 1.0


def _name_key(role, name):
    return format(zlib.crc32(f"{role}\n{fold(name)}".encode()), "08x")


def _role_score(want, got):
    if want == got:
        return 1.0
    return 0.6 if any(want in f and got in f for f in ROLE_FAMILIES) else 0.0


def _digits(s):
    return re.sub(r"\d+", "0", s)


def _name_score(fp, entry):
    a, b = fold(fp["name"]), fold(entry.name)
    if not a and not b:
        return None
    if a == b:
        return 1.0
    if fp.get("name_clipped") or entry.name_clipped:  # only the first 150 characters were printed
        n = min(len(a), len(b)) - 1
        if n > 20 and a[:n] == b[:n]:
            return 1.0
    if _digits(a) == _digits(b):
        return 0.9  # "Cart (2)" and "Cart (5)"
    ta, tb = set(tokens(_digits(a))), set(tokens(_digits(b)))
    jac = len(ta & tb) / len(ta | tb) if ta | tb else 0.0
    ratio = SequenceMatcher(None, a, b).ratio()
    return 0.85 * max(jac, ratio if ratio >= 0.8 else 0.0)  # character similarity only counts for near-typos


def _hits(want, got, decay, shift):
    """How much of the recorded lines is still next to the candidate, as (weighted score, total weight)."""
    num = den = 0.0
    for i, line in enumerate(want):
        total_w = sum(line.values())
        if total_w <= 0:
            continue
        best = 0.0
        for j, other in enumerate(got):
            hit = sum(w for t, w in line.items() if t in other)
            if hit > 0:
                extra = sum(1 for t in other if t not in line)
                best = max(best, hit / (total_w + EXTRA * extra) * shift ** abs(i - j))
        w = (decay ** i) * max(line.values())
        num += w * best
        den += w
    return num, den


def _ctx_score(fp, state, entry, cache):
    if fp["twins"] > 1:
        twins = cache.get((entry.role, fold(entry.name), entry.frame))
        if twins is None:
            twins = cache[(entry.role, fold(entry.name), entry.frame)] = _twins(state, entry)
        if len(twins) > 1:
            if "counts" not in cache:
                cache["counts"] = _name_counts(state)
            before, after, _ = _row(state, entry, twins, cache["counts"], fp.get("label_after", False), fp.get("row_size") or None)
        else:  # the only one left: look as far as the recorded row reached
            step = 1 if fp.get("label_after") else -1
            before = _window(state, entry.line, step, len(fp["before"]))
            after = _window(state, entry.line, -step, len(fp["after"]))
        decay = shift = 1.0  # anywhere inside the row is fine
    else:
        before, after = _window(state, entry.line, -1, BEFORE + 2), _window(state, entry.line, +1, AFTER + 2)
        decay, shift = DECAY, SHIFT
    bn, bd = _hits(fp["before"], [set(l) for l in before], decay, shift)
    an, ad = _hits(fp["after"], [set(l) for l in after], decay, shift)
    den = SIDE * bd + (1 - SIDE) * ad
    if den < 0.05:
        return None  # nothing around the element told it apart when it was recorded
    return (SIDE * bn + (1 - SIDE) * an) / den


def _head_score(fp, entry):
    a, b = fold(fp["heading"]), fold(entry.heading)
    same_frame = 1.0 if fp["frame"].split("?")[0] == entry.frame.split("?")[0] else 0.3
    if not a and not b:
        return same_frame if (fp["frame"] or entry.frame) else None
    ta, tb = set(tokens(a)), set(tokens(b))
    return same_frame * (1.0 if a == b else (len(ta & tb) / len(ta | tb) if ta | tb else 0.0))


def _ord_score(fp, state, entry):
    if fp["twins"] <= 1:
        return None
    twins = _twins(state, entry)
    if twins.index(entry) != fp["ordinal"]:
        return 0.0
    return 1.0 if len(twins) == fp["twins"] else 0.5


def components(fp, state):
    """[(entry, {role, name, ctx, head, ord})] for every candidate of a compatible role."""
    out, cache, known = [], {}, set(fp["known"])
    for e in state.entries:
        if e.hidden != fp.get("hidden", False):
            continue
        role = _role_score(fp["role"], e.role)
        if role == 0.0:
            continue
        out.append((e, {"role": role, "name": _name_score(fp, e), "ctx": _ctx_score(fp, state, e, cache),
                        "head": _head_score(fp, e), "ord": _ord_score(fp, state, e),
                        "new": _name_key(e.role, e.name) not in known}))
    return out


def total(c, p):
    num = den = 0.0
    for key, w in (("role", p.get("w_role", 1.0)), ("name", p["w_name"]), ("ctx", p["w_ctx"]), ("head", p["w_head"]), ("ord", p["w_ord"])):
        if c[key] is not None:
            num += w * c[key]
            den += w
    return num / den if den else 0.0


class Match:
    """entry is None when the lookup refused; status then says why."""

    def __init__(self, entry, status, confidence=0.0, margin=0.0, parts=None, runner_up=None):
        self.entry, self.status, self.confidence, self.margin = entry, status, confidence, margin
        self.parts, self.runner_up = parts or {}, runner_up

    @property
    def id(self):
        return self.entry.id if self.entry else None

    def as_dict(self):
        return {"status": self.status, "id": self.id, "confidence": round(self.confidence, 3),
                "margin": round(self.margin, 3),
                "match": format_entry(self.entry), "runner_up": format_entry(self.runner_up),
                "parts": {k: (None if v is None else round(v, 3)) for k, v in self.parts.items()}}

    def __repr__(self):
        return f"Match({self.status}, id={self.id}, conf={self.confidence:.2f}, margin={self.margin:.2f})"


def format_entry(e):
    return None if e is None else f'{e.role} "{e.name}"' + (f" under {e.heading!r}" if e.heading else "")


def decide(comps, fp, p=PARAMS):
    """Pick a candidate or refuse. Separate from `components` so thresholds can be tuned cheaply."""
    if not comps:
        return Match(None, "not_found")
    ranked = sorted(((total(c, p), -i, e, c) for i, (e, c) in enumerate(comps)), key=lambda t: (-t[0], -t[1]))
    exact = [r for r in ranked if r[3]["name"] is not None and r[3]["name"] >= 0.99 and r[3]["role"] == 1.0]
    if fp["twins"] == 1 and len(exact) == 1:
        # It was the only control with this role and name, and exactly one still is. A look-alike
        # in a better position ("Sun Hat 2" where "Sun Hat" used to be) must not outrank it.
        best, _, entry, c = exact[0]
        others = [r for r in ranked if r[2] is not entry]
        second, runner = (others[0][0], others[0][2]) if others else (0.0, None)
        # Under the same heading almost any surroundings will do. Under another heading it has
        # to bring its surroundings along, or it is a copy elsewhere and the original is gone.
        bar = p["solo_ctx"] if c["head"] is None or c["head"] >= 0.99 else p["ctx_min"]
        if c["ctx"] is not None and c["ctx"] < bar:
            return Match(None, "context_lost", best, best - second, c, runner)
        return Match(entry, "found", best, best - second, c, runner)
    best, _, entry, c = ranked[0]
    second = ranked[1][0] if len(ranked) > 1 else 0.0
    runner = ranked[1][2] if len(ranked) > 1 else None
    margin = best - second
    refuse = lambda why: Match(None, why, best, margin, c, runner)
    if (c["name"] or 0.0) < 0.9:
        # A different label is accepted only as a relabel: some of the old words, the same place,
        # and a name the recorded page did not already use for something else. "Remove" turning
        # into "Delete" is refused: from the listing alone it cannot be told from a new feature
        # that replaced the old one.
        if (c["name"] or 0.0) < p["rename_name"] or not c["new"] or c["ctx"] is None or c["ctx"] < p["rename_ctx"]:
            return refuse("not_found" if best < p["threshold"] else "renamed_unsure")
        return refuse("ambiguous") if margin < p["margin"] else Match(entry, "found", best, margin, c, runner)
    if best < p["threshold"]:
        return refuse("not_found")
    if margin < p["margin"]:
        return refuse("ambiguous")
    if (fp["twins"] > 1 or len(exact) > 1) and (c["ctx"] is None or c["ctx"] < p["ctx_min"]):
        # Several controls share this role and name, so only the surroundings say which one was
        # meant. If the recorded surroundings are gone, the element was probably removed.
        return refuse("context_lost")
    return Match(entry, "found", best, margin, c, runner)


def find(fp, state, params=PARAMS):
    """Locate the fingerprinted control in a fresh State. Returns a Match; .id is None on refusal."""
    if fp.get("v") != VERSION:
        raise ValueError(f"fingerprint version {fp.get('v')!r}, expected {VERSION}")
    if state.truncated:
        # The listing was cut in the middle (bu.state(max=...)): the element may be in the cut part,
        # and a look-alike in the visible part must not win by default.
        return Match(None, "state_truncated")
    return decide(components(fp, state), fp, params)


def main(argv):
    if len(argv) == 2 and argv[0] == "compile":
        from transcript import compile_plan
        with open(argv[1], encoding="utf-8") as f:
            print(json.dumps(compile_plan(json.load(f)), indent=1, ensure_ascii=False))
    elif len(argv) == 3 and argv[0] == "find":
        with open(argv[1], encoding="utf-8") as f:
            fp = json.load(f)
        with open(argv[2], encoding="utf-8") as f:
            m = find(fp, parse_state(f.read()))
        print(json.dumps(m.as_dict(), ensure_ascii=False))
        sys.exit(0 if m.entry else 1)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
