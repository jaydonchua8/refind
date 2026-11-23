"""refind: fingerprint a web element once, find it again after the page changes.

CLI:  python refind.py record page.html '/html[1]/body[1]/...' > fp.json
      python refind.py find fp.json new_page.html
"""
import json
import re
import sys
from difflib import SequenceMatcher
from html.parser import HTMLParser

# ---------- a tiny DOM ----------
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
SKIP = {"script", "style", "noscript", "template"}


class Node:
    __slots__ = ("tag", "attrs", "parent", "children", "text")

    def __init__(self, tag, attrs=None, parent=None):
        self.tag, self.attrs, self.parent = tag, dict(attrs or {}), parent
        self.children, self.text = [], ""

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()

    def full_text(self):
        return " ".join(n.text for n in self.walk() if n.text).strip()

    def index(self):
        """1-based position among same-tag siblings (XPath style)."""
        if self.parent is None:
            return 1
        same = [c for c in self.parent.children if c.tag == self.tag]
        return same.index(self) + 1

    def xpath(self):
        parts, n = [], self
        while n is not None and n.tag != "#root":
            parts.append(f"{n.tag}[{n.index()}]")
            n = n.parent
        return "/" + "/".join(reversed(parts))

    def path(self):
        tags, n = [], self.parent
        while n is not None and n.tag != "#root":
            tags.append(n.tag)
            n = n.parent
        return tags[::-1]


class _Builder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("#root")
        self.cur, self.skip = self.root, 0

    def handle_starttag(self, tag, attrs):
        if tag in SKIP:
            self.skip += 1
            return
        if self.skip:
            return
        node = Node(tag, {k: (v or "") for k, v in attrs}, self.cur)
        self.cur.children.append(node)
        if tag not in VOID:
            self.cur = node

    def handle_endtag(self, tag):
        if tag in SKIP:
            self.skip = max(0, self.skip - 1)
            return
        n = self.cur
        while n is not None and n.tag != tag:
            n = n.parent
        if n is not None and n.parent is not None:
            self.cur = n.parent

    def handle_data(self, data):
        if not self.skip and data.strip():
            self.cur.text = (self.cur.text + " " + " ".join(data.split())).strip()


def parse(html):
    b = _Builder()
    b.feed(html)
    b.close()
    return b.root


def by_xpath(root, xpath):
    node = root
    for part in xpath.strip("/").split("/"):
        tag, _, idx = part.partition("[")
        i = int(idx.rstrip("]")) if idx else 1
        same = [c for c in node.children if c.tag == tag]
        if i > len(same):
            return None
        node = same[i - 1]
    return node


# ---------- fingerprint and match ----------
# Attributes that describe what an element *is* tend to survive redesigns.
# Attributes that describe how it *looks* (class, style, generated ids) do not.
STABLE = {"name": 3.0, "aria-label": 3.0, "data-testid": 4.0, "role": 1.5, "type": 1.5,
          "placeholder": 2.5, "href": 2.0, "for": 2.0, "title": 2.0, "alt": 2.0, "value": 1.0}
GENERATED = re.compile(r"(^[a-z]{1,3}-?[0-9a-f]{5,}$)|([0-9]{4,})|(^css-)|(^sc-)|(__[A-Za-z0-9]{5}$)")
INTERACTIVE = {"a", "button", "input", "select", "textarea", "label", "summary", "option"}


def _tokens(s):
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def _jaccard(a, b):
    if not a and not b:
        return None  # no evidence either way
    return len(a & b) / len(a | b)


def _stable_classes(node):
    return {c for c in node.attrs.get("class", "").split() if not GENERATED.search(c)}


def _neighbors(node):
    if node.parent is None:
        return set()
    out = set()
    for sib in node.parent.children:
        if sib is not node:
            out |= _tokens(sib.full_text()[:60])
    return out


def fingerprint(node):
    """A JSON-serialisable description of `node` that does not rely on its position."""
    eid = node.attrs.get("id", "")
    return {
        "tag": node.tag,
        "id": "" if GENERATED.search(eid) else eid,
        "classes": sorted(_stable_classes(node)),
        "attrs": {k: v for k, v in node.attrs.items() if k in STABLE},
        "text": node.full_text()[:120],
        "path": node.path(),
        "index": node.index(),
        "neighbors": sorted(_neighbors(node)),
        "xpath": node.xpath(),
    }


def score(fp, node):
    """Similarity in [0, 1]. Signals with no evidence on either side are left out of the average."""
    if node.tag != fp["tag"]:
        return 0.0
    num = den = 0.0

    def add(weight, value):
        nonlocal num, den
        if value is not None:
            num += weight * value
            den += weight

    if fp["id"]:  # a matching id is strong evidence; a changed id is only weak evidence against
        same = node.attrs.get("id") == fp["id"]
        add(4.0 if same else 1.5, 1.0 if same else 0.0)
    for k, w in STABLE.items():
        want = fp["attrs"].get(k)
        if want is not None:
            got = node.attrs.get(k)
            add(w, 0.0 if got is None else SequenceMatcher(None, want, got).ratio())
    add(3.0, _jaccard(_tokens(fp["text"]), _tokens(node.full_text()[:120])))
    add(1.0, _jaccard(set(fp["classes"]), _stable_classes(node)))
    add(1.5, SequenceMatcher(None, fp["path"], node.path()).ratio())
    add(1.5, _jaccard(set(fp["neighbors"]), _neighbors(node)))
    add(0.5, 1.0 if node.index() == fp["index"] else 0.0)
    return num / den if den else 0.0


class Match:
    def __init__(self, node, confidence, margin, method):
        self.node, self.confidence, self.margin, self.method = node, confidence, margin, method

    def __repr__(self):
        where = self.node.xpath() if self.node else None
        return f"Match({self.method}, conf={self.confidence:.2f}, margin={self.margin:.2f}, {where})"


def find(fp, root, threshold=0.55, min_margin=0.04):
    """Locate the fingerprinted element in a new DOM.

    Returns a Match. `node` is None when nothing clears `threshold`, or when the top two
    candidates are closer than `min_margin`. Refusing an ambiguous match matters more than
    finding one: a replayed agent that clicks the wrong button is worse than one that stops.
    """
    ranked = sorted(((score(fp, n), i, n) for i, n in enumerate(root.walk()) if n.tag == fp["tag"]),
                    key=lambda t: (-t[0], t[1]))
    if not ranked or ranked[0][0] < threshold:
        return Match(None, ranked[0][0] if ranked else 0.0, 0.0, "not_found")
    best = ranked[0]
    margin = best[0] - ranked[1][0] if len(ranked) > 1 else best[0]
    if margin < min_margin:
        return Match(None, best[0], margin, "ambiguous")
    return Match(best[2], best[0], margin, "healed" if best[2].xpath() != fp["xpath"] else "exact")


# ---------- CLI ----------
def main(argv):
    if len(argv) == 3 and argv[0] == "record":
        node = by_xpath(parse(open(argv[1]).read()), argv[2])
        if node is None:
            sys.exit("no element at that xpath")
        print(json.dumps(fingerprint(node), indent=1))
    elif len(argv) == 3 and argv[0] == "find":
        m = find(json.load(open(argv[1])), parse(open(argv[2]).read()))
        print(json.dumps({"status": m.method, "confidence": round(m.confidence, 3),
                          "margin": round(m.margin, 3), "xpath": m.node.xpath() if m.node else None}))
        sys.exit(0 if m.node else 1)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
