"""Read and write the page listing that browser-use 1.0 prints in ultrafast mode.

In browser-use 1.0 the model sees a page as the text that `bu.state()` prints
(browser_use/ax.py, class AxHelpers, method state):

    [state] <title> | <url>
    ## <heading>
    <visible text>
    [<backendNodeId>] <role> "<name>" = "<value>" <states>  options: a | b | ... n more
    --- frame <url> ---
    [/state]

and every `bu.click(id)` / `bu.type(id, text)` prints one line:

    [ok] click [<id>] <role> "<name>" -> navigated to <url> | page changed | no change

This module parses both, and can write them back out in the same shape so that the
synthetic simulator in sim.py produces text in the same format. The formatting rules
(clip lengths, which part is JSON-quoted, the order of the state words) were read from
ax.py at commit 753d629 (version 1.0.8). Nothing here imports or runs browser-use.

The printed format is not injective. Two cases cannot be told apart from the text alone
and are documented in the README: a name that itself contains `" = "`, and page text that
looks exactly like a control line.
"""
import json
import re

STATE_WORDS = ("unchecked", "checked", "expanded", "collapsed", "selected", "disabled", "focused")
NAME_CLIP, VALUE_CLIP, DESCRIBE_CLIP, SHOWN_OPTIONS = 150, 100, 60, 5
ELLIPSIS = "…"
OUTCOMES = ("navigated", "page changed", "no change")

_HEAD = re.compile(r"^(\*?)\[(\d+)\] ([A-Za-z][A-Za-z0-9]*)(.*)$")
_LOADING = re.compile(r" \| loading: (\d+)$")
_CUT = re.compile(r"^\.\.\. (\d+) lines cut here \(bu\.state\(max=30000\) or js\(\.\.\.\)\)$")
_OK = re.compile(r"^\[ok\] (click|type) \[(\d+)\](.*) -> (navigated to (\S+)(?: \((.*)\))?|page changed|no change)$")
_FAILED = re.compile(r"(click|type) \[(\d+)\](.*?) failed: (.*)$")


class StateError(ValueError):
    """The text is not a complete [state] ... [/state] block."""


def fold(s):
    """Lowercase with every whitespace run as one space (ax.py fold): for matching, never printed."""
    return re.sub(r"\s+", " ", s or "").strip().lower()


def clip(s, n):
    return s[:n] + ELLIPSIS if len(s) > n else s


def quote(s):
    return json.dumps(s, ensure_ascii=False)


class Entry:
    """One control line. `line` is its index in State.lines."""

    __slots__ = ("id", "role", "name", "value", "states", "options", "more", "new", "hidden",
                 "heading", "frame", "line")

    def __init__(self, id, role, name="", value="", states=(), options=(), more=0, new=False, hidden=False):
        self.id, self.role, self.name, self.value = id, role, name, value
        self.states, self.options, self.more = tuple(states), tuple(options), more
        self.new, self.hidden = new, hidden
        self.heading, self.frame, self.line = "", "", -1

    @property
    def name_clipped(self):
        return len(self.name) == NAME_CLIP + 1 and self.name.endswith(ELLIPSIS)

    def describe(self):
        """What an action line calls this element (ax.py _describe)."""
        return f"[{self.id}] {self.role}" + (f' "{clip(self.name, DESCRIBE_CLIP)}"' if self.name else "")

    def __repr__(self):
        return f"Entry({format_control(self)!r})"


class State:
    """A parsed listing. lines: (kind, text, entry) with kind in heading|text|control|frame|cut|note."""

    def __init__(self, title, url, loading=0):
        self.title, self.url, self.loading = title, url, loading
        self.dialogs, self.lines, self.entries, self.cut = [], [], [], 0

    def by_id(self, id):
        return next((e for e in self.entries if e.id == id), None)

    @property
    def truncated(self):
        return self.cut > 0


def format_control(e):
    """The line ax.py prints for a control (hidden file inputs have their own shape)."""
    if e.hidden:
        return f'[{e.id}] file "{e.name}" (hidden)'
    shown = list(e.options[:SHOWN_OPTIONS])
    more = e.more or max(0, len(e.options) - SHOWN_OPTIONS)
    return (("*" if e.new else "") + f"[{e.id}] {e.role}"
            + (f' "{clip(e.name, NAME_CLIP)}"' if e.name else "")
            + (f" = {quote(clip(e.value, VALUE_CLIP))}" if e.value else "")
            + (" " + " ".join(e.states) if e.states else "")
            + ("  options: " + " | ".join(shown) + (f" | {ELLIPSIS} {more} more" if more else "") if shown else ""))


def parse_control(line):
    """An Entry, or None when the line is not a control line (it is then page text)."""
    m = _HEAD.match(line)
    if not m:
        return None
    star, id, role, rest = m.group(1) == "*", int(m.group(2)), m.group(3), m.group(4)
    if rest == "":
        return Entry(id, role, new=star)
    if role == "file" and not star and rest.startswith(' "') and rest.endswith('" (hidden)'):
        return Entry(id, role, rest[2:-len('" (hidden)')], hidden=True)
    options, more = [], 0
    # Names, values and option labels have their whitespace runs collapsed, so two spaces only
    # ever appear before "options:".
    if "  options: " in rest:
        rest, _, tail = rest.partition("  options: ")
        options = tail.split(" | ")
        mm = re.fullmatch(ELLIPSIS + r" (\d+) more", options[-1]) if options else None
        if mm:
            more, options = int(mm.group(1)), options[:-1]
    last = rest.rfind('"')
    tail = rest[last + 1:] if last >= 0 else rest
    words = tail.split(" ")[1:] if tail else []
    if tail and (not tail.startswith(" ") or any(w not in STATE_WORDS for w in words)):
        return None
    if [STATE_WORDS.index(w) for w in words] != sorted({STATE_WORDS.index(w) for w in words}):
        return None  # ax.py always prints the state words once each, in this order
    head = rest[:last + 1] if last >= 0 else ""
    name = value = ""
    if head:
        cut = _value_start(head)
        if cut is not None:
            value, head = json.loads(head[cut + 3:]), head[:cut]
        if head:
            if len(head) < 3 or not head.startswith(' "') or not head.endswith('"'):
                return None
            name = head[2:-1]
    if not (name or value or words or options):
        return None
    return Entry(id, role, name, value, words, options, more, star)


def _value_start(head):
    """Index of the ' = ' that begins the JSON-quoted value ending `head`, or None.

    The name is printed raw and may contain quotes; the value is JSON, so its inner quotes are
    escaped. The leftmost ' = "' whose remainder is one valid JSON string is the value.
    """
    at = head.find(' = "')
    while at >= 0:
        if at == 0 or head[at - 1] == '"':
            try:
                if isinstance(json.loads(head[at + 3:]), str):
                    return at
            except ValueError:
                pass
        at = head.find(' = "', at + 1)
    return None


def format_state(title, url, lines, loading=0, dialogs=()):
    """lines: strings already in printed form (format_control, '## heading', text, '--- frame u ---')."""
    header = f"[state] {title} | {url}" + (f" | loading: {loading}" if loading else "")
    return "\n".join([header, *dialogs, *lines, "[/state]"])


def parse_state(text):
    """Parse one block. Raises StateError when the header or the closing line is missing."""
    rows = text.strip("\n").split("\n")
    if not rows or not rows[0].startswith("[state] "):
        raise StateError("no [state] header")
    if rows[-1] != "[/state]":
        raise StateError("no [/state] line: the output was cut")
    header = rows[0][len("[state] "):]
    loading = 0
    m = _LOADING.search(header)
    if m:
        loading, header = int(m.group(1)), header[:m.start()]
    title, sep, url = header.rpartition(" | ")
    if not sep:
        raise StateError("header has no ' | ' between title and url")
    st = State(title, url, loading)
    body = rows[1:-1]
    while body and body[0].startswith("[dialog "):
        st.dialogs.append(body.pop(0))
    heading = frame = ""
    for row in body:
        if row.startswith("## "):
            heading = row[3:]
            st.lines.append(("heading", heading, None))
        elif row.startswith("--- frame ") and row.endswith(" ---"):
            frame, heading = row[len("--- frame "):-4], ""
            st.lines.append(("frame", frame, None))
        elif _CUT.match(row):
            st.cut += int(_CUT.match(row).group(1))
            st.lines.append(("cut", row, None))
        elif row.startswith("(") and row.endswith(")") and len(body) == 1:
            st.lines.append(("note", row, None))
        else:
            e = parse_control(row)
            if e is None:
                st.lines.append(("text", row, None))
            else:
                e.heading, e.frame, e.line = heading, frame, len(st.lines)
                st.lines.append(("control", row, e))
                st.entries.append(e)
    return st


def split_output(text):
    """Cut one cell's output into events in printed order.

    Yields ('state', State) | ('state_error', message) | ('action', dict) | ('failed', dict).
    An action dict has op, id, rest (the text between the id and ' -> '), outcome, url.
    """
    rows = text.split("\n")
    i = 0
    while i < len(rows):
        row = rows[i]
        if row.startswith("[state] "):
            try:
                j = rows.index("[/state]", i + 1)
            except ValueError:
                yield ("state_error", "a [state] block has no [/state] line: the output was cut")
                return
            try:
                yield ("state", parse_state("\n".join(rows[i:j + 1])))
            except StateError as e:
                yield ("state_error", str(e))
            i = j + 1
            continue
        m = _OK.match(row)
        if m:
            kind = "navigated" if m.group(4).startswith("navigated") else m.group(4)
            yield ("action", {"op": m.group(1), "id": int(m.group(2)), "rest": m.group(3),
                              "outcome": kind, "url": m.group(5) or "", "nav_error": m.group(6) or ""})
        else:
            f = _FAILED.search(row) if " failed: " in row else None
            if f:
                yield ("failed", {"op": f.group(1), "id": int(f.group(2)), "reason": f.group(4)})
        i += 1


def parse_type_detail(detail):
    """The detail of a type line after `[id] role "name"`.

    Returns {text, complete, enter}. text is None for a hidden password; complete is False when
    the printed text was clipped at 60 characters, so the full text has to come from the code.
    """
    out = {"text": None, "complete": False, "enter": detail.endswith(" +Enter")}
    if not detail.startswith(" = "):
        return None
    body = detail[3:]
    if body.startswith("<"):
        return out if re.match(r"<\d+ characters, hidden>", body) else None
    try:
        text, _ = json.JSONDecoder().raw_decode(body)
    except ValueError:
        return None
    if not isinstance(text, str):
        return None
    out["text"] = text
    out["complete"] = not (len(text) == DESCRIBE_CLIP + 1 and text.endswith(ELLIPSIS))
    return out
