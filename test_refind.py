import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refind import parse, by_xpath, fingerprint, find

OLD = """<html><body><form><label for="e">Email</label><input id="e" name="email" type="email">
<button type="submit" class="btn primary">Sign in</button><button type="button" class="btn">Sign in with SSO</button></form></body></html>"""
NEW = """<html><body><div class="cookie"><button>Accept</button></div><div class="x9f2"><form>
<div><label for="q1">Email</label><input id="q1" name="email" type="email"></div>
<div><button type="button" class="k2">Sign in with SSO</button><button type="submit" class="k1">Sign in</button></div></form></div></body></html>"""


def node_with(root, tag, text):
    return next(n for n in root.walk() if n.tag == tag and n.full_text() == text)


class RefindTests(unittest.TestCase):
    def test_xpath_roundtrip(self):
        root = parse(OLD)
        for n in root.walk():
            if n.tag != "#root":
                self.assertIs(by_xpath(root, n.xpath()), n)

    def test_unchanged_page_is_exact(self):
        root = parse(OLD)
        b = node_with(root, "button", "Sign in")
        m = find(fingerprint(b), parse(OLD))
        self.assertEqual(m.method, "exact")
        self.assertEqual(m.node.xpath(), b.xpath())

    def test_heals_after_redesign_where_xpath_breaks(self):
        old, new = parse(OLD), parse(NEW)
        b = node_with(old, "button", "Sign in")
        fp = fingerprint(b)
        stale = by_xpath(new, fp["xpath"])
        self.assertTrue(stale is None or stale.full_text() != "Sign in")  # the recorded xpath no longer works
        m = find(fp, new)
        self.assertEqual(m.method, "healed")
        self.assertEqual(m.node.full_text(), "Sign in")
        self.assertEqual(m.node.attrs["type"], "submit")

    def test_input_found_by_name_when_id_changes(self):
        fp = fingerprint(node_with(parse(OLD), "label", "Email").parent.children[1])
        self.assertEqual(find(fp, parse(NEW)).node.attrs["name"], "email")

    def test_refuses_when_element_is_gone(self):
        fp = fingerprint(node_with(parse(OLD), "button", "Sign in with SSO"))
        gone = parse("<html><body><button>Delete account</button></body></html>")
        m = find(fp, gone)
        self.assertIsNone(m.node)
        self.assertEqual(m.method, "not_found")

    def test_refuses_ambiguous_duplicates(self):
        fp = fingerprint(node_with(parse("<html><body><p>x</p><button>Remove</button></body></html>"), "button", "Remove"))
        twins = parse("<html><body><div><button>Remove</button></div><div><button>Remove</button></div></body></html>")
        m = find(fp, twins)
        self.assertIsNone(m.node)
        self.assertEqual(m.method, "ambiguous")

    def test_generated_ids_and_classes_are_ignored(self):
        root = parse('<html><body><button id="r1a2b3c4d" class="css-9zx1 save">Save</button></body></html>')
        fp = fingerprint(next(n for n in root.walk() if n.tag == "button"))
        self.assertEqual(fp["id"], "")
        self.assertEqual(fp["classes"], ["save"])


if __name__ == "__main__":
    unittest.main()
