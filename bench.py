"""Benchmark: can a recorded element still be found after the page is redesigned?

For every interactive element on each fixture page we record three locators (absolute XPath,
a CSS selector, a refind fingerprint), apply k random mutations to the page, and check which
locators still resolve to the same element. Ground truth is free: mutations edit the tree in
place, so the original Node object is the right answer.
"""
import glob, os, random, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refind import parse, by_xpath, fingerprint, find
from refind import INTERACTIVE, Node


def body(root):
    return next(n for n in root.walk() if n.tag == "body")

def m_banner(root, t, rng):
    b = body(root); d = Node("div", {"class": "banner"}, b); d.text = "We use cookies"
    btn = Node("button", {}, d); btn.text = "Accept"; d.children.append(btn); b.children.insert(0, d)

def m_wrap(root, t, rng):
    p = t.parent; g = p.parent
    if g is None: return
    w = Node("div", {"class": "wrap"}, g); g.children[g.children.index(p)] = w; w.children.append(p); p.parent = w

def m_classes(root, t, rng):
    for n in root.walk():
        if "class" in n.attrs:
            n.attrs["class"] = " ".join("c" + format(rng.getrandbits(24), "x") for _ in n.attrs["class"].split())

def m_ids(root, t, rng):
    for n in root.walk():
        if "id" in n.attrs: n.attrs["id"] = "r" + format(rng.getrandbits(28), "x")

def m_reorder(root, t, rng):
    p = t.parent
    if p.parent is not None: rng.shuffle(p.parent.children)

def m_insert_sibling(root, t, rng):
    p = t.parent; s = Node(t.tag, {}, p); s.text = "New"; p.children.insert(p.children.index(t), s)

def m_retext(root, t, rng):
    if t.text: t.text = t.text + " now"

MUTATIONS = [m_banner, m_wrap, m_classes, m_ids, m_reorder, m_insert_sibling, m_retext]


def css(node):
    if node.attrs.get("id"): return ("id", node.attrs["id"])
    return ("cls", node.tag, frozenset(node.attrs.get("class", "").split()))

def css_resolve(root, sel):
    if sel[0] == "id": hits = [n for n in root.walk() if n.attrs.get("id") == sel[1]]
    else: hits = [n for n in root.walk() if n.tag == sel[1] and sel[2] and sel[2] <= set(n.attrs.get("class", "").split())]
    return hits[0] if len(hits) == 1 else None


def run(trials=40, seed=7):
    rng = random.Random(seed)
    pages = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "page_*.html")))
    out = {}
    for k in (1, 2, 3):
        c = dict(n=0, xpath=0, css=0, refind=0, refind_wrong=0, refind_refused=0)
        for path in pages:
            html = open(path).read()
            targets = [n.xpath() for n in parse(html).walk() if n.tag in INTERACTIVE]
            for xp in targets:
                for _ in range(trials):
                    root = parse(html); t = by_xpath(root, xp)
                    fp, sel = fingerprint(t), css(t)
                    for m in rng.sample(MUTATIONS, k): m(root, t, rng)
                    c["n"] += 1
                    c["xpath"] += by_xpath(root, xp) is t
                    c["css"] += css_resolve(root, sel) is t
                    got = find(fp, root).node
                    c["refind"] += got is t
                    c["refind_wrong"] += got is not None and got is not t
                    c["refind_refused"] += got is None
        out[k] = c
    return out


if __name__ == "__main__":
    res = run()
    print(f"{'mutations':>9} {'trials':>7} {'xpath':>7} {'css':>7} {'refind':>7} {'wrong':>7} {'refused':>8}")
    for k, c in res.items():
        p = lambda x: f"{100 * c[x] / c['n']:.1f}%"
        print(f"{k:>9} {c['n']:>7} {p('xpath'):>7} {p('css'):>7} {p('refind'):>7} {p('refind_wrong'):>7} {p('refind_refused'):>8}")
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_results.json"), "w"), indent=1)
