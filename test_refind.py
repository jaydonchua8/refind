import json
import os
import random
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim
from axstate import (Entry, StateError, format_control, format_state, parse_control, parse_state,
                     parse_type_detail, split_output)
from refind import fingerprint, find, PARAMS
from replay import handoff_prompt, replay, same_page
from transcript import TranscriptError, compile_plan, read_code

# The expected output in browser-use's own test (tests/ci/test_ultrafast.py, test_state_format,
# commit 753d629), with its [N] placeholders given numbers. It is the one piece of real
# browser-use output in this repository; everything else is synthetic.
UPSTREAM = """[state] Demo | http://127.0.0.1:8000/form
## Book a trip
Pick a city and a date.
[11] textbox "City" = "Bern"
[12] checkbox "Flexible" checked
[13] select "Country" = "Austria" collapsed  options: Austria | Canada | Chile | Denmark | Estonia | … 2 more
[14] Date "Day"
[15] button "Details" collapsed
[16] button "Unavailable" disabled
[17] link "Next page"
[/state]"""


def listing(*lines, title="T", url="https://x.test/p"):
    """A State from lines written by hand: 'h:Heading', 't:text', or (id, role, name[, states])."""
    out = []
    for l in lines:
        if isinstance(l, str):
            out.append("## " + l[2:] if l.startswith("h:") else l[2:])
        else:
            out.append(format_control(Entry(l[0], l[1], l[2], states=l[3] if len(l) > 3 else ())))
    return parse_state(format_state(title, url, out))


def team(*people, start=10, extra=()):
    lines, i = ["h:Team members", (start, "textbox", "Search members"), (start + 1, "button", "Invite member")], start + 2
    for who in people:
        lines += [f"t:{who}@example.test Member", (i, "button", "Edit"), (i + 1, "button", "Remove")]
        i += 2
    return listing(*lines, *extra)


def id_after(state, text, name):
    """The id of the control called `name` in the row that starts with the text line `text`."""
    at = next(i for i, l in enumerate(state.lines) if l[0] == "text" and l[1].startswith(text))
    return next(l[2].id for l in state.lines[at + 1:] if l[0] == "control" and l[2].name == name)


class ListingFormat(unittest.TestCase):
    def test_upstream_example_parses(self):
        st = parse_state(UPSTREAM)
        self.assertEqual((st.title, st.url), ("Demo", "http://127.0.0.1:8000/form"))
        self.assertEqual([(e.id, e.role, e.name, e.value, e.states) for e in st.entries], [
            (11, "textbox", "City", "Bern", ()), (12, "checkbox", "Flexible", "", ("checked",)),
            (13, "select", "Country", "Austria", ("collapsed",)), (14, "Date", "Day", "", ()),
            (15, "button", "Details", "", ("collapsed",)), (16, "button", "Unavailable", "", ("disabled",)),
            (17, "link", "Next page", "", ())])
        country = st.by_id(13)
        self.assertEqual((country.options, country.more), (("Austria", "Canada", "Chile", "Denmark", "Estonia"), 2))
        self.assertTrue(all(e.heading == "Book a trip" for e in st.entries))
        self.assertEqual(st.lines[1], ("text", "Pick a city and a date.", None))

    def test_upstream_example_round_trips(self):
        st = parse_state(UPSTREAM)
        self.assertEqual([format_control(e) for e in st.entries],
                         [l for l in UPSTREAM.split("\n") if l.startswith("[1")])

    def test_quotes_and_equals_inside_names_and_values(self):
        e = parse_control('[5] button "Say "hi" now" = "a \\"b\\" = \\"c\\"" focused')
        self.assertEqual((e.name, e.value, e.states), ('Say "hi" now', 'a "b" = "c"', ("focused",)))
        e = parse_control('*[6] textbox = "only a value"')
        self.assertEqual((e.name, e.value, e.new), ("", "only a value", True))
        e = parse_control('[7] link "x = "q""')
        self.assertEqual((e.name, e.value), ('x = "q"', ""))

    def test_page_text_is_not_taken_for_a_control(self):
        for text in ("[2] see the footnote", "[12] Button \"x\" maybe", "[3] button \"a\" clicked", "[x] button", "(12) button \"a\"",
                     '[4] button "a" focused checked'):
            self.assertIsNone(parse_control(text), text)
        st = listing("t:[2] see the footnote", (9, "button", "Go"))
        self.assertEqual([e.id for e in st.entries], [9])

    def test_fuzzed_lines_round_trip(self):
        rng = random.Random(5)
        alphabet = ['a', 'B', '"', '=', ' = ', '|', '(', ')', 'é', '\\', '[1]', 'checked', ' ']
        text = lambda n: " ".join("".join(rng.choice(alphabet) for _ in range(rng.randrange(1, n))).split())
        n = 0
        for _ in range(3000):
            name, value = text(8) if rng.random() < 0.8 else "", text(8) if rng.random() < 0.5 else ""
            if '" = "' in name or not (name or value):
                continue  # the documented ambiguity: a name that itself contains '" = "'
            states = [w for w in ("checked", "expanded", "focused") if rng.random() < 0.3]
            options = [f"opt{i}" for i in range(rng.randrange(0, 8))] if rng.random() < 0.3 else []
            e = Entry(rng.randrange(1, 99999), rng.choice(["button", "link", "textbox", "select", "Date"]), name, value, states, options,
                      new=rng.random() < 0.2)
            back = parse_control(format_control(e))
            self.assertIsNotNone(back, format_control(e))
            self.assertEqual((back.id, back.role, back.name, back.value, back.states, back.new),
                             (e.id, e.role, e.name, e.value, e.states, e.new), format_control(e))
            self.assertEqual(len(back.options) + back.more, len(options))
            n += 1
        self.assertGreater(n, 2000)

    def test_header_extras_frames_hidden_files_and_cut(self):
        st = parse_state("\n".join([
            "[state] A | B | https://x.test/a?b=1 | loading: 2", '[dialog alert] "Hello" (accepted)',
            '[31] file "cv" (hidden)', '[32] button "Top"', "... 40 lines cut here (bu.state(max=30000) or js(...))",
            "--- frame https://x.test/inner ---", '[33] button "Inner button"', "[/state]"]))
        self.assertEqual((st.title, st.url, st.loading, st.dialogs), ("A | B", "https://x.test/a?b=1", 2, ['[dialog alert] "Hello" (accepted)']))
        self.assertTrue(st.by_id(31).hidden)
        self.assertEqual((st.by_id(32).frame, st.by_id(33).frame), ("", "https://x.test/inner"))
        self.assertEqual(st.cut, 40)
        self.assertTrue(st.truncated)

    def test_incomplete_blocks_are_errors(self):
        with self.assertRaises(StateError):
            parse_state('[state] T | https://x.test/\n[1] button "a"')
        with self.assertRaises(StateError):
            parse_state("hello")
        with self.assertRaises(StateError):
            parse_state("[state] no separator\n[/state]")
        self.assertEqual([k for k, _ in split_output('[state] T | u\n[1] button "a"')], ["state_error"])

    def test_action_lines(self):
        out = "\n".join(['[ok] click [7] button "Go" -> page changed', '[ok] type [8] textbox "Query" = "Bern" +Enter -> navigated to http://h/next?q=Bern',
                         '[ok] click [9] link "Dead" -> navigated to chrome-error://chromewebdata/ (net::ERR_FAILED)',
                         '[ok] type [3] textbox "a -> b" = "x" -> no change', "noise",
                         'Error: RuntimeError: click [999999] failed: CDP -32000: No node with given id found'])
        ev = list(split_output(out))
        self.assertEqual([k for k, _ in ev], ["action", "action", "action", "action", "failed"])
        self.assertEqual((ev[1][1]["op"], ev[1][1]["id"], ev[1][1]["outcome"], ev[1][1]["url"]), ("type", 8, "navigated", "http://h/next?q=Bern"))
        self.assertEqual(ev[2][1]["nav_error"], "net::ERR_FAILED")
        self.assertEqual(ev[3][1]["rest"], ' textbox "a -> b" = "x"')
        self.assertEqual(ev[4][1]["id"], 999999)

    def test_type_detail(self):
        self.assertEqual(parse_type_detail(' = "Bern" +Enter'), {"text": "Bern", "complete": True, "enter": True})
        self.assertEqual(parse_type_detail(' = "' + "x" * 60 + '…"')["complete"], False)
        self.assertEqual(parse_type_detail(" = <12 characters, hidden>")["text"], None)
        self.assertEqual(parse_type_detail(' = "A" (field now shows "B")')["text"], "A")
        self.assertIsNone(parse_type_detail(""))


class Matching(unittest.TestCase):
    def test_fingerprint_is_json_and_holds_no_id(self):
        st = team("ana", "bo", "cy")
        fp = json.loads(json.dumps(fingerprint(st, id_after(st, "bo@", "Remove"))))
        self.assertNotIn("id", fp)
        self.assertEqual((fp["role"], fp["name"], fp["twins"], fp["ordinal"]), ("button", "Remove", 3, 1))
        self.assertEqual(fp["before"][-1]["bo@example.test"], 1.0)   # the row's own label
        self.assertEqual(fp["before"][0]["edit"], 0.0)                # on every row: tells nothing
        with self.assertRaises(KeyError):
            fingerprint(st, 424242)
        with self.assertRaises(ValueError):
            find(dict(fp, v=1), st)

    def test_same_page_new_ids(self):
        old, new = team("ana", "bo", "cy"), team("ana", "bo", "cy", start=900)
        for e in old.entries:
            m = find(fingerprint(old, e.id), new)
            self.assertEqual(m.id, e.id + 890, e)

    def test_first_row_ignores_the_toolbar_above_the_table(self):
        st = team("ana", "bo", "cy")
        fp = fingerprint(st, id_after(st, "ana@", "Edit"))
        self.assertEqual([sorted(l) for l in fp["before"]], [["ana@example.test", "member"]])

    def test_duplicates_follow_their_row_when_rows_move(self):
        old, new = team("ana", "bo", "cy"), team("dee", "cy", "ana", "bo", start=500)
        for who in ("ana", "bo", "cy"):
            for name in ("Edit", "Remove"):
                m = find(fingerprint(old, id_after(old, who, name)), new)
                self.assertEqual((m.status, m.id), ("found", id_after(new, who, name)), (who, name))

    def test_refuses_when_its_row_is_gone(self):
        old, new = team("ana", "bo", "cy"), team("ana", "cy", "dee", start=500)
        m = find(fingerprint(old, id_after(old, "bo@", "Remove")), new)
        self.assertIsNone(m.id)
        self.assertIn(m.status, ("context_lost", "not_found"))

    def test_refuses_when_only_one_twin_is_left_and_it_is_another_row(self):
        old = team("ana", "bo")
        self.assertIsNone(find(fingerprint(old, id_after(old, "bo@", "Remove")), team("ana", start=300)).id)
        self.assertEqual(find(fingerprint(old, id_after(old, "ana@", "Remove")), team("ana", start=300)).id, 303)

    def test_identical_rows_with_no_label_are_refused(self):
        old = listing("h:Slots", (1, "button", "Book"), (2, "button", "Book"), (3, "button", "Book"))
        new = listing("h:Slots", (7, "button", "Book"), (8, "button", "Book"), (9, "button", "Book"))
        m = find(fingerprint(old, 2), new)
        self.assertIsNone(m.id)

    def test_exact_unique_name_beats_a_lookalike_in_its_old_place(self):
        old = listing("h:Results", (1, "link", "Sun Hat"), "t:$20", (2, "button", "Add to cart"),
                      (3, "link", "Rain Shell"), "t:$30", (4, "button", "Add to cart"))
        new = listing("h:Results", (11, "link", "Sun Hat Kids"), "t:$25", (12, "button", "Add to cart"),
                      (13, "link", "Sun Hat"), "t:$20", (14, "button", "Add to cart"),
                      (15, "link", "Rain Shell"), "t:$30", (16, "button", "Add to cart"))
        self.assertEqual(find(fingerprint(old, 1), new).id, 13)
        self.assertEqual(find(fingerprint(old, 2), new).id, 14)

    def test_rows_whose_label_follows_the_controls(self):
        def files(order, start):
            lines, i = ["h:Files"], start
            for name in order:
                lines += [(i, "button", "Download"), (i + 1, "button", "Delete"), f"t:{name}.pdf 2 MB"]
                i += 2
            return listing(*lines, "h:Footer", (99, "link", "Help"))
        old, new = files(["a-report", "b-invoice", "c-notes"], 10), files(["b-invoice", "d-new", "a-report"], 50)
        self.assertTrue(fingerprint(old, 12)["label_after"])
        got = [find(fingerprint(old, i), new).id for i in (10, 11, 12, 13, 14, 15)]
        self.assertEqual(got, [54, 55, 50, 51, None, None])   # c-notes is gone: refused, not sent to d-new

    def test_row_that_lost_its_button_is_not_merged_into_the_next_row(self):
        def shop(b_button, start):
            return listing("h:Results", (start, "link", "Sun Hat"), "t:$20", (start + 1, "button", "Add to cart"),
                           (start + 2, "link", "Rain Shell"), "t:$30", (start + 3, "button", b_button),
                           (start + 4, "link", "Day Pack"), "t:$40", (start + 5, "button", "Add to cart"))
        old, new = shop("Add to cart", 1), shop("Notify me", 51)
        self.assertEqual(fingerprint(old, 4)["row_size"], 2)
        self.assertIsNone(find(fingerprint(old, 4), new).id)      # Rain Shell is out of stock now
        self.assertEqual(find(fingerprint(old, 6), new).id, 56)   # Day Pack's button is not Rain Shell's
        self.assertEqual(find(fingerprint(old, 2), new).id, 52)

    def test_counter_in_name(self):
        old = listing((1, "link", "Cart (2)"), (2, "link", "Sign in"))
        self.assertEqual(find(fingerprint(old, 1), listing((8, "link", "Cart (5)"), (9, "link", "Sign in"))).id, 8)

    def test_role_swap_between_button_and_link(self):
        old = listing("h:Cart", "t:Subtotal $40", (1, "button", "Checkout"), (2, "link", "Continue shopping"))
        new = listing("h:Cart", "t:Subtotal $40", (5, "link", "Checkout"), (6, "link", "Continue shopping"))
        self.assertEqual(find(fingerprint(old, 1), new).id, 5)
        self.assertIsNone(find(fingerprint(old, 1), listing("h:Cart", (5, "textbox", "Checkout"))).id)

    def test_relabel_needs_shared_words_the_same_place_and_a_new_name(self):
        old = listing("h:Sign in", (1, "textbox", "Email"), (2, "textbox", "Password"), (3, "button", "Sign in"),
                      (4, "button", "Sign in with SSO"), (5, "link", "Forgot password?"))
        relabelled = listing("h:Sign in", (11, "textbox", "Email"), (12, "textbox", "Password"), (13, "button", "Log in"),
                             (14, "button", "Sign in with SSO"), (15, "link", "Forgot password?"))
        self.assertEqual(find(fingerprint(old, 3), relabelled).id, 13)
        # The button is gone. "Sign in with SSO" shares words and sits right there, but it was
        # already on the recorded page as a different element.
        removed = listing("h:Sign in", (11, "textbox", "Email"), (12, "textbox", "Password"),
                          (14, "button", "Sign in with SSO"), (15, "link", "Forgot password?"))
        m = find(fingerprint(old, 3), removed)
        self.assertIsNone(m.id)
        # No shared word: could be a relabel or a different feature in the same spot. Refused.
        other = listing("h:Sign in", (11, "textbox", "Email"), (12, "textbox", "Password"), (13, "button", "Continue"),
                        (14, "button", "Sign in with SSO"), (15, "link", "Forgot password?"))
        self.assertIsNone(find(fingerprint(old, 3), other).id)

    def test_a_copy_elsewhere_does_not_win_and_a_lone_copy_elsewhere_is_refused(self):
        old = listing("h:Team", (1, "textbox", "Search members"), (2, "button", "Invite member"), "t:ana@example.test Admin", (3, "button", "Edit"))
        both = listing("h:Shortcuts", "t:Popular right now", (20, "button", "Invite member"),
                       "h:Team", (21, "textbox", "Search members"), (22, "button", "Invite member"), "t:ana@example.test Admin", (23, "button", "Edit"))
        self.assertEqual(find(fingerprint(old, 2), both).id, 22)
        only_copy = listing("h:Shortcuts", "t:Popular right now", (20, "button", "Invite member"),
                            "h:Team", (21, "textbox", "Search members"), "t:ana@example.test Admin", (23, "button", "Edit"))
        m = find(fingerprint(old, 2), only_copy)
        self.assertEqual((m.id, m.status), (None, "context_lost"))

    def test_cut_listing_is_refused(self):
        old = listing((1, "button", "Save"))
        cut = parse_state('[state] T | https://x.test/p\n[5] button "Save"\n... 12 lines cut here (bu.state(max=30000) or js(...))\n[/state]')
        self.assertEqual(find(fingerprint(old, 1), cut).status, "state_truncated")

    def test_other_frame_is_a_different_twin_group(self):
        text = "\n".join(["[state] T | https://x.test/p", '[1] button "Next month"', "--- frame https://x.test/cal ---",
                          "November", '[2] button "Next month"', "[/state]"])
        old, new = parse_state(text), parse_state(text.replace("[1]", "[8]").replace("[2]", "[9]"))
        self.assertEqual(find(fingerprint(old, 2), new).id, 9)
        self.assertEqual(find(fingerprint(old, 1), new).id, 8)


def run_cell(code, output, n=[0]):
    n[0] += 1
    return [{"kind": "response", "parts": [{"part_kind": "tool-call", "tool_name": "python", "tool_call_id": f"c{n[0]}", "args": {"code": code}}]},
            {"kind": "request", "parts": [{"part_kind": "tool-return", "tool_name": "python", "tool_call_id": f"c{n[0]}", "content": [output]}]}]


PAGE = '[state] T | https://x.test/p\n[5] button "Go"\n[6] textbox "Query"\n[/state]'


class Compiling(unittest.TestCase):
    def setUp(self):
        self.site = sim.make_admin(random.Random(1))
        self.messages, self.keys = sim.record(self.site, self.site.tasks[0], random.Random(2))

    def test_synthetic_run_compiles_completely(self):
        plan = compile_plan(self.messages)
        kinds = [s["kind"] for s in plan["steps"]]
        self.assertEqual(kinds[0], "goto")
        self.assertEqual(kinds[-1], "deliver")
        self.assertEqual(sum(k in ("click", "type") for k in kinds), len(self.keys))
        self.assertEqual(plan["counts"]["opaque"], 0)
        typed = next(s for s in plan["steps"] if s["kind"] == "type")
        self.assertEqual((typed["text"], typed["enter"], typed["expect"]["outcome"]), ("new.hire@example.test", False, "page changed"))
        self.assertNotIn("id", typed["fp"])
        json.dumps(plan)

    def test_accepted_input_shapes(self):
        want = compile_plan(self.messages)
        self.assertEqual(compile_plan({"version": 1, "messages": self.messages}), want)
        as_text = json.loads(json.dumps(self.messages))
        for m in as_text:
            for p in m["parts"]:
                if p["part_kind"] == "tool-call":
                    p["args"] = json.dumps(p["args"])
                if p["part_kind"] == "tool-return":
                    p["content"] = p["content"][0]
        self.assertEqual(compile_plan(as_text), want)
        with self.assertRaises(TranscriptError):
            compile_plan({"nope": 1})
        with self.assertRaises(TranscriptError):
            compile_plan(["x"])

    def test_code_reader(self):
        ops, why = read_code("new_tab('https://x.test/p'); wait_for_load()\ns = bu.state(max=30000)\nbu.click(5)\nprint(bu.type(6, 'a', enter=True))\nbu.type(6, 'b', True)")
        self.assertEqual([o["kind"] for o in ops], ["goto", "look", "click", "type", "type"])
        self.assertEqual((ops[3]["enter"], ops[4]["enter"]), (True, True))
        for code in ("for i in ids:\n    bu.click(i)", "bu.click(n)", "js('document.title')", "bu.click(5) if x else None",
                     "cdp('DOM.focus', backendNodeId=5)", "bu.type(6, text)", "bu.click(5", "click_at_xy(3, 4)", "bu.type(6, 'a', enter=maybe)"):
            self.assertIsNone(read_code(code)[0], code)

    def opaque_reason(self, *cells):
        msgs = [m for c in cells for m in run_cell(*c)]
        steps = compile_plan(msgs)["steps"]
        return [s["reason"] for s in steps if s["kind"] == "opaque"], steps

    def test_cells_that_must_not_be_replayed(self):
        look = ("new_tab('https://x.test/p')\nbu.state()", PAGE)
        ok = '[ok] click [5] button "Go" -> page changed\n' + PAGE
        for cell, word in [
            (("ids = [5]\nfor i in ids:\n    bu.click(i)", ok), "not a plain helper call"),
            (("rows = js('document.title')", "x"), "js() is not replayed"),
            (("bu.click(5)", 'Error: RuntimeError: click [5] button "Go" failed: Target covered by [9] dialog "Cookies"; nothing was clicked\n' + PAGE), "failed in the recorded run"),
            (("bu.click(77)", '[ok] click [77] -> page changed\n' + PAGE), "not in the last page listing"),
            (("bu.click(5)", '[ok] click [6] textbox "Query" -> page changed\n' + PAGE), "do not match the code"),
            (("bu.click(5)", PAGE), "no [ok] line"),
            (("bu.click(5)", ok + "\n" + ok), "more actions than the code"),
            (("bu.type(6, '<secret:pw>')", '[ok] type [6] textbox "Query" = "<secret:pw>" -> page changed\n' + PAGE), "redacted secret"),
            (("bu.type(6, 'abc')", '[ok] type [6] textbox "Query" = "xyz" -> page changed\n' + PAGE), "differs from the text in the code"),
            (("bu.click(5)", '[ok] click [5] button "Go" -> page changed\n[state] T | https://x.test/p\n[5] button "Go"'), "incomplete"),
        ]:
            reasons, _ = self.opaque_reason(look, cell)
            self.assertEqual(len(reasons), 1, cell)
            self.assertIn(word, reasons[0], cell)

    def test_id_used_after_navigation_without_a_new_listing(self):
        look = ("new_tab('https://x.test/p')\nbu.state()", PAGE)
        nav = ("bu.click(5)\nbu.click(6)", '[ok] click [5] button "Go" -> navigated to https://x.test/q\n[ok] click [6] textbox "Query" -> no change\n' + PAGE)
        reasons, _ = self.opaque_reason(look, nav)
        self.assertIn("after a navigation", reasons[0])
        first = ("bu.click(5)", '[ok] click [5] button "Go" -> page changed\n' + PAGE)
        reasons, _ = self.opaque_reason(("bu.click(5)", '[ok] click [5] button "Go" -> page changed'))
        self.assertIn("before any page listing", reasons[0])
        reasons, steps = self.opaque_reason(look, first)
        self.assertEqual((reasons, steps[-1]["kind"], steps[-1]["target"]), ([], "click", 'button "Go"'))

    def test_clipped_text_comes_from_the_code_and_other_tools_are_opaque(self):
        long = "x" * 80
        look = ("new_tab('https://x.test/p')\nbu.state()", PAGE)
        typed = (f"bu.type(6, {long!r})", f'[ok] type [6] textbox "Query" = "{"x" * 60}…" -> page changed\n' + PAGE)
        reasons, steps = self.opaque_reason(look, typed)
        self.assertEqual((reasons, steps[-1]["text"]), ([], long))
        msgs = run_cell(*look) + [{"kind": "response", "parts": [{"part_kind": "tool-call", "tool_name": "send_email", "tool_call_id": "t", "args": {"to": "a"}}]},
                                  {"kind": "response", "parts": [{"part_kind": "tool-call", "tool_name": "finish", "tool_call_id": "f", "args": {}}]}]
        self.assertEqual([s["kind"] for s in compile_plan(msgs)["steps"]], ["goto", "opaque", "deliver"])


class Replaying(unittest.TestCase):
    def setUp(self):
        self.site = sim.make_admin(random.Random(1))
        self.task = self.site.tasks[1]   # edit member r2, toggle 2FA, rename, save, remove member r1
        self.messages, self.keys = sim.record(self.site, self.task, random.Random(2))
        self.plan = compile_plan(self.messages)

    def go(self, site, seed=9, **kw):
        b = sim.Browser(site, random.Random(seed))
        return replay(self.plan, sim.FakeBu(b), goto=b.goto_url, **kw), b

    def changed(self, path, edit):
        pages = dict(self.site.pages)
        pages[path] = sim.copy.deepcopy(pages[path])
        edit(pages[path])
        return sim.Site(self.site.name, self.site.host, pages, self.site.tasks)

    def test_same_site_new_session(self):
        r, b = self.go(self.site)
        self.assertEqual((r["status"], b.hits), ("completed", self.keys))
        recorded = [int(c) for m in self.messages for p in m["parts"] if p["part_kind"] == "tool-call" and p["tool_name"] == "python"
                    for c in __import__("re").findall(r"bu\.\w+\((\d+)", p["args"]["code"])]
        self.assertTrue(set(recorded).isdisjoint(x["id"] for x in r["log"] if "id" in x))  # no recorded id was reused

    def test_reordered_and_grown_table(self):
        def edit(page):
            rng = random.Random(4)
            for op in ("banner", "reorder", "insert", "noise", "numbers"):
                self.assertTrue(sim.DRIFTS[op](page, rng))
        r, b = self.go(self.changed("/team", edit))
        self.assertEqual((r["status"], b.hits), ("completed", self.keys))

    def test_removed_row_stops_before_acting(self):
        def edit(page):
            page.items = [it for it in page.items if not (it.get("list") == "members" and it.get("row") == "r1")]
        r, b = self.go(self.changed("/team", edit))
        self.assertEqual(r["status"], "handoff")
        self.assertEqual((r["handoff"]["acted"], b.hits), (False, self.keys[:-1]))
        self.assertIn(r["handoff"]["reason"], ("context_lost", "not_found"))
        text = handoff_prompt(self.plan, r)
        self.assertIn('click button "Remove"', text)
        self.assertIn("was not attempted", text)

    def test_toggle_already_in_the_target_state(self):
        def edit(page):
            sim._flip(next(c for c in page.controls() if c["key"] == "edit.twofa"))
        r, b = self.go(self.changed("/edit", edit))
        self.assertEqual((r["handoff"]["reason"], r["handoff"]["acted"]), ("state_differs", False))
        self.assertEqual(b.hits, self.keys[:1])

    def test_disabled_control(self):
        def edit(page):
            next(c for c in page.controls() if c["key"] == "edit.save")["states"].append("disabled")
        r, b = self.go(self.changed("/edit", edit))
        self.assertEqual((r["handoff"]["reason"], b.hits), ("disabled", self.keys[:3]))

    def test_outcome_differs_is_reported_as_already_sent(self):
        def edit(page):
            next(c for c in page.controls() if c["key"] == "edit.save")["effect"] = ("change",)
        r, b = self.go(self.changed("/edit", edit))
        self.assertEqual((r["handoff"]["reason"], r["handoff"]["acted"]), ("outcome_differs", True))
        self.assertEqual(b.hits, self.keys[:4])
        self.assertIn("already sent", handoff_prompt(self.plan, r))

    def test_landing_on_another_page(self):
        def edit(page):
            next(c for c in page.controls() if c["key"] == "members.r2.edit")["effect"] = ("nav", "/invite")
        r, b = self.go(self.changed("/team", edit))
        self.assertEqual((r["handoff"]["reason"], len(b.hits)), ("outcome_differs", 1))
        self.assertTrue(same_page("https://a.test/x/?q=1#f", "https://a.test/x"))
        self.assertFalse(same_page("https://a.test/x", "https://a.test/y"))

    def test_missing_page(self):
        pages = {k: v for k, v in self.site.pages.items() if k != "/team"}
        r, b = self.go(sim.Site(self.site.name, self.site.host, pages, []))
        self.assertEqual((r["handoff"]["reason"], b.hits), ("not_found", []))

    def test_page_text_forged_as_a_control_cannot_be_clicked(self):
        def edit(page):
            page.items = [it for it in page.items if it.get("key") != "edit.save"]
            page.items.append(sim.T('[99999] button "Save changes"'))
        r, b = self.go(self.changed("/edit", edit))
        self.assertEqual((r["handoff"]["reason"], b.hits), ("action_failed", self.keys[:3]))
        self.assertIn("No node with given id", r["handoff"]["detail"])

    def test_no_goto_callable_and_unreadable_listing(self):
        b = sim.Browser(self.site, random.Random(3))
        self.assertEqual(replay(self.plan, sim.FakeBu(b))["handoff"]["reason"], "no_goto")

        class Cut(sim.FakeBu):
            def state(self, max=3000):
                return super().state().rsplit("\n", 1)[0]
        r = replay(self.plan, Cut(b), goto=b.goto_url)
        self.assertEqual((r["handoff"]["reason"], b.hits), ("state_unreadable", []))

    def test_opaque_steps(self):
        plan = json.loads(json.dumps(self.plan))
        plan["steps"].insert(2, {"kind": "opaque", "cell": 99, "reason": "js() is not replayed", "code": "title = js('document.title')", "id_free": True})
        b = sim.Browser(self.site, random.Random(3))
        r = replay(plan, sim.FakeBu(b), goto=b.goto_url)
        self.assertEqual((r["handoff"]["reason"], r["handoff"]["step"], len(b.hits)), ("opaque_cell", 2, 1))
        ran = []
        b = sim.Browser(self.site, random.Random(3))
        r = replay(plan, sim.FakeBu(b), goto=b.goto_url, run_cell=ran.append)
        self.assertEqual((r["status"], ran, b.hits), ("completed", ["title = js('document.title')"], self.keys))
        plan["steps"][2].update(code="cdp('DOM.focus', backendNodeId=1400)", id_free=False)
        b = sim.Browser(self.site, random.Random(3))
        self.assertEqual(replay(plan, sim.FakeBu(b), goto=b.goto_url, run_cell=ran.append)["handoff"]["reason"], "opaque_cell")

        def boom(code):
            raise NameError("rows")
        plan["steps"][2].update(code="print(rows)", id_free=True)
        b = sim.Browser(self.site, random.Random(3))
        self.assertEqual(replay(plan, sim.FakeBu(b), goto=b.goto_url, run_cell=boom)["handoff"]["reason"], "cell_failed")


class SimulatorAndBenchmark(unittest.TestCase):
    def test_every_scripted_task_records_and_replays_on_its_own_site(self):
        for make in sim.TRAIN_SITES + sim.HELD_OUT_SITES:
            for seed in range(3):
                rng = random.Random(seed)
                site = make(rng)
                for task in site.tasks:
                    messages, keys = sim.record(site, task, rng)
                    plan = compile_plan(json.loads(sim.dumps(messages)))
                    self.assertEqual(plan["counts"]["opaque"], 0)
                    b = sim.Browser(site, random.Random(seed + 50))
                    r = replay(plan, sim.FakeBu(b), goto=b.goto_url)
                    self.assertEqual((r["status"], b.hits), ("completed", keys), (site.name, seed))

    def test_drift_keeps_keys_unique_and_every_operation_applies_somewhere(self):
        applied = set()
        for make in sim.TRAIN_SITES + sim.HELD_OUT_SITES:
            rng = random.Random(8)
            site = make(rng)
            for page in site.pages.values():
                for name in sim.DRIFTS:
                    changed, done = sim.drift_page(page, rng, 1, {name: 1})
                    applied.update(done)
                    keys = [c["key"] for c in changed.controls()]
                    self.assertEqual(len(keys), len(set(keys)), (site.name, name))
        self.assertEqual(applied, set(sim.DRIFTS))

    def test_benchmark_runs_and_refind_is_safer_than_nth(self):
        import bench
        res = bench.run_elements("held_out", range(1))
        mine, nth = res["refind"]["all"], res["role+name, nth"]["all"]
        self.assertGreater(mine["present"], 200)
        self.assertLess(mine["wrong"] + mine["absent_wrong"], nth["wrong"] + nth["absent_wrong"])
        self.assertGreater(mine["correct"], nth["correct"])
        self.assertEqual(res["recorded id"]["all"]["correct"], 0)


class CommandLine(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, os.path.join(HERE, "refind.py"), *args], capture_output=True, text=True, cwd=HERE)

    def test_compile_and_find(self):
        import tempfile
        p = self.run_cli("compile", os.path.join(HERE, "sample_transcript.json"))
        self.assertEqual(p.returncode, 0, p.stderr)
        plan = json.loads(p.stdout)
        step = next(s for s in plan["steps"] if s["kind"] == "click")
        with tempfile.TemporaryDirectory() as d:
            fp, page = os.path.join(d, "fp.json"), os.path.join(d, "state.txt")
            with open(fp, "w") as f:
                json.dump(step["fp"], f)
            site = sim.make_helpdesk(random.Random(3))
            b = sim.Browser(site, random.Random(77))
            b.goto_url(step["page"]["url"])
            with open(page, "w") as f:
                f.write(sim.FakeBu(b).state())
            found = self.run_cli("find", fp, page)
            self.assertEqual(found.returncode, 0, found.stderr)
            self.assertEqual(json.loads(found.stdout)["status"], "found")
            with open(page, "w") as f:
                f.write('[state] Empty | https://desk.example.test/tickets\n[1] button "Unrelated"\n[/state]')
            self.assertEqual(self.run_cli("find", fp, page).returncode, 1)
        self.assertNotEqual(self.run_cli().returncode, 0)


if __name__ == "__main__":
    unittest.main()
