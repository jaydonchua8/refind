"""SYNTHETIC sites and a stand-in for browser-use's `bu` helpers. Everything here is made up.

Nothing in this file talks to a browser. It exists so the matcher and the replay can be measured
offline with known ground truth:

- four small synthetic sites (two for tuning, two held out), built from page models;
- `Browser` + `FakeBu`: prints page listings and action lines in the text format of
  browser_use/ax.py, hands out fresh ids on every page load (as Chrome backendNodeIds are), and
  records which element each action really hit;
- `drift`: the changes a site goes through between a recording and a replay;
- `record`: runs a scripted task and writes the transcript JSON that transcript.py reads.

Every control carries a hidden `key` that survives drift. The matcher never sees it; the
benchmark uses it to tell a correct match from a wrong one.
"""
import copy
import json
import random
import re

from axstate import Entry, format_control, format_state, clip, quote, DESCRIBE_CLIP

PEOPLE = "dana omar priya lucas mei tomas aisha jonas keiko ravi elena sam nora felix ines hugo".split()
ROLES = ["Admin", "Member", "Billing", "Viewer"]
PRODUCTS = ["Trail Runner", "Road Racer", "Court Classic", "Hiking Boot", "Summit Jacket", "Rain Shell", "Day Pack",
            "Trek Pole", "Camp Stove", "Wool Sock", "Sun Hat", "River Sandal", "Alpine Glove", "Bivy Sack"]
ISSUES = ["Printer offline", "VPN drops", "Password reset", "Laptop will not boot", "Email bounce", "Badge not working",
          "Slow wifi", "Monitor flicker", "License expired", "Shared drive full", "Calendar sync", "Phone no dial tone"]
HOTELS = ["Harbor Inn", "Pine Lodge", "Hotel Aurora", "Casa Verde", "The Meridian", "Old Mill House", "Lakeview Suites",
          "Station Hotel", "Villa Rosa", "Summit Chalet", "Garden Court", "Blue Door Hostel"]
ROOMS = ["King room", "Twin room", "Family suite", "Studio", "Loft"]

# A redesign's relabels. Train and held-out sites use different labels, so different rows apply.
SYNONYMS = {"Sign in": "Log in", "Add to cart": "Add to bag", "Remove": "Delete", "Checkout": "Proceed to checkout",
            "Search products": "Search", "Invite member": "Invite", "Send invite": "Send invitation", "Edit": "Manage",
            "Save changes": "Save", "Place order": "Buy now", "Next page": "Next", "Resend": "Send again",
            "Assign to me": "Take ticket", "Close": "Resolve", "Send reply": "Reply", "View deal": "See availability",
            "Reserve": "Book now", "Search": "Find stays", "Confirm booking": "Complete booking", "Wake": "Unsnooze",
            "Filter tickets": "Search tickets", "Save": "Add to wishlist", "Close ticket": "Mark resolved"}
HEADINGS = {"Products": "All products", "Team members": "People", "Pending invites": "Invitations", "Your cart": "Cart",
            "Open tickets": "Inbox", "Snoozed": "Snoozed tickets", "Filters": "Refine", "Payment": "Payment details",
            "Rooms": "Choose a room", "Guest details": "Your details", "Shipping": "Delivery"}


def H(text):
    return {"k": "h", "text": text}


def T(text, row=None, lst=None):
    return {"k": "t", "text": text, "row": row, "list": lst}


def C(key, role, name, value="", states=(), options=(), effect=None, enter=None, row=None, lst=None, password=False):
    return {"k": "c", "key": key, "role": role, "name": name, "value": value, "states": list(states),
            "options": list(options), "effect": effect, "enter": enter, "row": row, "list": lst, "password": password}


class Page:
    def __init__(self, title, items, factories=None):
        self.title, self.items, self.factories = title, items, factories or {}
        self.fresh = 0  # counter for rows added by drift
        self.relabel, self.rerole = {}, set()  # site-wide relabels chosen by drift, applied at the end

    def controls(self):
        return [it for it in self.items if it["k"] == "c"]

    def rows(self, lst):
        """[(row id, [items])] in page order, and the index where the list starts."""
        out, start = [], None
        for i, it in enumerate(self.items):
            if it.get("list") == lst:
                start = i if start is None else start
                if not out or out[-1][0] != it["row"]:
                    out.append((it["row"], []))
                out[-1][1].append(it)
        return out, start

    def set_rows(self, lst, rows, start):
        rest = [it for it in self.items if it.get("list") != lst]
        keep_before = len([it for it in self.items[:start] if it.get("list") != lst])
        self.items = rest[:keep_before] + [it for _, items in rows for it in items] + rest[keep_before:]


class Site:
    def __init__(self, name, host, pages, tasks):
        self.name, self.host, self.pages, self.tasks = name, host, pages, tasks

    def url(self, path):
        return f"https://{self.host}{path}"


# ---------- site generators ----------
def _rowlist(page_items, factories, lst, n, make):
    factories[lst] = make
    for i in range(n):
        page_items.extend(make(f"r{i}"))


def make_shop(rng):
    names = rng.sample(PRODUCTS, 9)
    it = iter(names + [f"{n} {rng.choice(['Pro', 'Lite', 'Kids', 'XL', '2'])}" for n in names])

    def product(lst):
        def make(rid, name=None):
            name = name or next(it)
            return [C(f"{lst}.{rid}.open", "link", name, effect=("nav", "/product"), row=rid, lst=lst),
                    T(f"${rng.randrange(19, 240)}", rid, lst),
                    C(f"{lst}.{rid}.add", "button", "Add to cart", effect=("change",), row=rid, lst=lst),
                    C(f"{lst}.{rid}.save", "button", "Save for later", effect=("change",), row=rid, lst=lst)]
        return make

    def cart_row(rid, name=None):
        name = name or next(it)
        return [T(f"{name} ${rng.randrange(19, 240)}", rid, "cart"),
                C(f"cart.{rid}.qty", "spinbutton", "Quantity", value="1", effect=("change",), row=rid, lst="cart"),
                C(f"cart.{rid}.remove", "button", "Remove", effect=("change",), row=rid, lst="cart")]

    top = lambda p: [C(f"{p}.search", "searchbox", "Search products", enter=("nav", "/search")),
                     C(f"{p}.cart", "link", f"Cart ({rng.randrange(1, 6)})", effect=("nav", "/cart")),
                     C(f"{p}.login", "link", "Sign in", effect=("nav", "/login"))]
    pages = {}
    for path, title, head, lst, n in (("/", "Acme Outdoor", "Products", "products", rng.randrange(3, 6)),
                                      ("/search", "Search | Acme Outdoor", "Results", "results", rng.randrange(2, 5))):
        f, items = {}, [H("Acme Outdoor"), *top(lst), H("Filters"),
                        C(f"{lst}.instock", "checkbox", "In stock", states=["unchecked"], effect=("toggle",)),
                        C(f"{lst}.sale", "checkbox", "On sale", states=["unchecked"], effect=("toggle",)),
                        C(f"{lst}.sort", "select", "Sort by", value="Relevance", states=["collapsed"],
                          options=["Relevance", "Price: low to high", "Price: high to low", "Newest"], effect=("change",)),
                        H(head)]
        _rowlist(items, f, lst, n, product(lst))
        items.append(C(f"{lst}.next", "link", "Next page", effect=("nav", path)))
        pages[path] = Page(title, items, f)
    f, items = {}, [H("Your cart")]
    _rowlist(items, f, "cart", rng.randrange(2, 5), cart_row)
    items += [T(f"Subtotal ${rng.randrange(100, 900)}"), C("cart.checkout", "button", "Checkout", effect=("nav", "/checkout")),
              C("cart.back", "link", "Continue shopping", effect=("nav", "/"))]
    pages["/cart"] = Page("Cart | Acme Outdoor", items, f)
    pages["/checkout"] = Page("Checkout | Acme Outdoor", [
        H("Shipping"), C("co.name", "textbox", "Full name", effect=("change",)), C("co.addr", "textbox", "Address", effect=("change",)),
        C("co.city", "textbox", "City", effect=("change",)),
        C("co.country", "select", "Country", value="Austria", states=["collapsed"],
          options=["Austria", "Canada", "Chile", "Denmark", "Estonia", "France", "Ghana"], effect=("change",)),
        H("Payment"), C("co.card", "textbox", "Card number", effect=("change",)),
        C("co.savecard", "checkbox", "Save this card", states=["unchecked"], effect=("toggle",)),
        C("co.place", "button", "Place order", effect=("nav", "/done")), C("co.back", "link", "Back to cart", effect=("nav", "/cart"))])
    pages["/login"] = Page("Sign in | Acme Outdoor", [
        H("Sign in"), C("login.email", "textbox", "Email", effect=("change",)),
        C("login.pw", "textbox", "Password", effect=("change",), password=True),
        C("login.remember", "checkbox", "Remember me", states=["unchecked"], effect=("toggle",)),
        C("login.submit", "button", "Sign in", effect=("nav", "/")), C("login.sso", "button", "Sign in with SSO", effect=("nav", "/")),
        C("login.forgot", "link", "Forgot password?", effect=("nav", "/login"))])
    pages["/done"] = Page("Thank you | Acme Outdoor", [H("Order placed"), T("We sent a receipt to your email.")])
    pages["/product"] = Page("Product | Acme Outdoor", [H("Product"), T("Details"),
                                                        C("product.add", "button", "Add to cart", effect=("change",))])
    tasks = [("/", [("type", "products.search", "trail shoes", True), ("click", "results.r1.add"), ("click", "results.cart"),
                    ("click", "cart.r0.remove"), ("click", "cart.checkout"), ("type", "co.name", "Sam Lee", False),
                    ("type", "co.city", "Urbana", False), ("type", "co.country", "Canada", False), ("click", "co.place")]),
             ("/", [("click", "products.login"), ("type", "login.email", "sam@example.test", False),
                    ("click", "login.remember"), ("click", "login.submit"), ("click", "products.instock"),
                    ("click", "products.r2.save")])]
    return Site("shop", "shop.example.test", pages, tasks)


def make_admin(rng):
    people = rng.sample(PEOPLE, len(PEOPLE))
    it = iter(people)

    def member(rid):
        who = next(it)
        return [T(f"{who}@example.test {rng.choice(ROLES)}", rid, "members"),
                C(f"members.{rid}.edit", "button", "Edit", effect=("nav", "/edit"), row=rid, lst="members"),
                C(f"members.{rid}.remove", "button", "Remove", effect=("change",), row=rid, lst="members")]

    def invite(rid):
        who = next(it)
        return [T(f"{who}@example.test invited {rng.randrange(1, 9)} days ago", rid, "invites"),
                C(f"invites.{rid}.resend", "button", "Resend", effect=("change",), row=rid, lst="invites"),
                C(f"invites.{rid}.revoke", "button", "Revoke", effect=("change",), row=rid, lst="invites")]

    f, items = {}, [H("Team members"), C("team.search", "textbox", "Search members", effect=("change",)),
                    C("team.invite", "button", "Invite member", effect=("nav", "/invite"))]
    _rowlist(items, f, "members", rng.randrange(3, 7), member)
    items.append(H("Pending invites"))
    _rowlist(items, f, "invites", rng.randrange(2, 4), invite)
    pages = {"/team": Page("Team | Northwind Admin", items, f),
             "/invite": Page("Invite | Northwind Admin", [
                 H("Invite a member"), C("inv.email", "textbox", "Email address", effect=("change",)),
                 C("inv.role", "select", "Role", value="Member", states=["collapsed"], options=ROLES, effect=("change",)),
                 C("inv.welcome", "checkbox", "Send welcome email", states=["checked"], effect=("toggle",)),
                 C("inv.send", "button", "Send invite", effect=("nav", "/team")), C("inv.cancel", "link", "Cancel", effect=("nav", "/team"))]),
             "/edit": Page("Edit member | Northwind Admin", [
                 H("Edit member"), C("edit.name", "textbox", "Display name", value="Sam", effect=("change",)),
                 C("edit.role", "select", "Role", value="Member", states=["collapsed"], options=ROLES, effect=("change",)),
                 C("edit.twofa", "switch", "Two-factor required", states=["unchecked"], effect=("toggle",)),
                 C("edit.save", "button", "Save changes", effect=("nav", "/team")),
                 C("edit.delete", "button", "Delete account", effect=("change",))])}
    tasks = [("/team", [("click", "team.invite"), ("type", "inv.email", "new.hire@example.test", False),
                        ("type", "inv.role", "Billing", False), ("click", "inv.send"), ("click", "invites.r1.resend")]),
             ("/team", [("click", "members.r2.edit"), ("click", "edit.twofa"), ("type", "edit.name", "Samira", False),
                        ("click", "edit.save"), ("click", "members.r1.remove")])]
    return Site("admin", "admin.example.test", pages, tasks)


def make_helpdesk(rng):
    issues = rng.sample(ISSUES, len(ISSUES))
    it = iter(issues)
    num = iter(rng.sample(range(4000, 4999), 40))

    def ticket(rid):
        return [C(f"open.{rid}.open", "link", f"#{next(num)} {next(it)}", effect=("nav", "/ticket"), row=rid, lst="open"),
                T(f"Opened by {rng.choice(PEOPLE)} {rng.randrange(2, 50)} hours ago", rid, "open"),
                C(f"open.{rid}.assign", "button", "Assign to me", effect=("change",), row=rid, lst="open"),
                C(f"open.{rid}.close", "button", "Close", effect=("change",), row=rid, lst="open")]

    def snoozed(rid):
        return [C(f"snz.{rid}.open", "link", f"#{next(num)} {next(it)}", effect=("nav", "/ticket"), row=rid, lst="snz"),
                C(f"snz.{rid}.wake", "button", "Wake", effect=("change",), row=rid, lst="snz")]

    f, items = {}, [H("Open tickets"), C("tk.filter", "searchbox", "Filter tickets", enter=("nav", "/tickets")),
                    C("tk.status", "select", "Status", value="Open", states=["collapsed"], options=["Open", "Pending", "Solved"], effect=("change",)),
                    C("tk.mine", "checkbox", "Only mine", states=["unchecked"], effect=("toggle",))]
    _rowlist(items, f, "open", rng.randrange(3, 7), ticket)
    items.append(H("Snoozed"))
    _rowlist(items, f, "snz", rng.randrange(2, 4), snoozed)
    files = iter(rng.sample(["screenshot", "error-log", "invoice", "floor-plan", "config", "trace", "photo", "notes"], 8))

    def attachment(rid):  # the one list here whose label comes AFTER its controls
        return [C(f"att.{rid}.get", "button", "Download", effect=("change",), row=rid, lst="att"),
                C(f"att.{rid}.del", "button", "Delete", effect=("change",), row=rid, lst="att"),
                T(f"{next(files)}.pdf {rng.randrange(20, 900)} KB", rid, "att")]

    f2, ticket_items = {}, [
        H("Ticket"), T("The device on floor 3 stopped responding this morning."),
        C("t.priority", "select", "Priority", value="Normal", states=["collapsed"], options=["Low", "Normal", "High"], effect=("change",)),
        C("t.reply", "textbox", "Reply", effect=("change",)),
        C("t.internal", "checkbox", "Internal note", states=["unchecked"], effect=("toggle",)),
        C("t.send", "button", "Send reply", effect=("change",)),
        C("t.close", "button", "Close ticket", effect=("nav", "/tickets")),
        C("t.back", "link", "All tickets", effect=("nav", "/tickets")), H("Attachments")]
    _rowlist(ticket_items, f2, "att", rng.randrange(2, 5), attachment)
    pages = {"/tickets": Page("Tickets | Deskly", items, f), "/ticket": Page("Ticket | Deskly", ticket_items, f2)}
    tasks = [("/tickets", [("click", "open.r1.assign"), ("click", "open.r1.open"), ("type", "t.priority", "High", False),
                           ("type", "t.reply", "Looking into it now.", False), ("click", "t.send"), ("click", "att.r1.get"), ("click", "t.close"),
                           ("click", "snz.r0.wake")]),
             ("/tickets", [("click", "tk.mine"), ("type", "tk.filter", "printer", True), ("click", "open.r2.close"),
                           ("click", "open.r0.open"), ("click", "t.internal"), ("type", "t.reply", "Escalated.", False),
                           ("click", "t.send")])]
    return Site("helpdesk", "desk.example.test", pages, tasks)


def make_booking(rng):
    hotels = rng.sample(HOTELS, len(HOTELS))
    it = iter(hotels)

    def stay(rid):
        return [C(f"stays.{rid}.open", "link", next(it), effect=("nav", "/stay"), row=rid, lst="stays"),
                T(f"${rng.randrange(80, 400)} per night {rng.randrange(30, 50) / 10} stars", rid, "stays"),
                C(f"stays.{rid}.deal", "button", "View deal", effect=("nav", "/stay"), row=rid, lst="stays"),
                C(f"stays.{rid}.save", "button", "Save", effect=("change",), row=rid, lst="stays")]

    rooms = iter(rng.sample(ROOMS, len(ROOMS)) + ["Annex room", "Attic room", "Garden room"])

    def room(rid):
        return [T(f"{next(rooms)} ${rng.randrange(90, 420)}", rid, "rooms"),
                C(f"rooms.{rid}.reserve", "button", "Reserve", effect=("nav", "/guest"), row=rid, lst="rooms")]

    f1, results = {}, [H("Stays"), C("res.sort", "select", "Sort", value="Recommended", states=["collapsed"],
                                     options=["Recommended", "Price", "Rating"], effect=("change",)),
                       C("res.breakfast", "checkbox", "Breakfast included", states=["unchecked"], effect=("toggle",))]
    _rowlist(results, f1, "stays", rng.randrange(3, 7), stay)
    f2, stay_items = {}, [H("Rooms"), T("Free cancellation on most rooms.")]
    _rowlist(stay_items, f2, "rooms", rng.randrange(2, 5), room)
    stay_items.append(C("stay.back", "link", "Back to results", effect=("nav", "/results")))
    pages = {"/": Page("Wayfare", [
                 H("Find a stay"), C("home.dest", "combobox", "Destination", effect=("change",)),
                 C("home.in", "Date", "Check in", effect=("change",)), C("home.out", "Date", "Check out", effect=("change",)),
                 C("home.guests", "spinbutton", "Guests", value="2", effect=("change",)),
                 C("home.flex", "checkbox", "Flexible dates", states=["unchecked"], effect=("toggle",)),
                 C("home.search", "button", "Search", effect=("nav", "/results"))]),
             "/results": Page("Results | Wayfare", results, f1),
             "/stay": Page("Stay | Wayfare", stay_items, f2),
             "/guest": Page("Guest details | Wayfare", [
                 H("Guest details"), C("g.first", "textbox", "First name", effect=("change",)),
                 C("g.last", "textbox", "Last name", effect=("change",)), C("g.email", "textbox", "Email", effect=("change",)),
                 C("g.insure", "switch", "Travel insurance", states=["unchecked"], effect=("toggle",)),
                 C("g.confirm", "button", "Confirm booking", effect=("nav", "/confirmed")),
                 C("g.back", "link", "Change room", effect=("nav", "/stay"))]),
             "/confirmed": Page("Confirmed | Wayfare", [H("Booking confirmed"), T("See you soon.")])}
    tasks = [("/", [("type", "home.dest", "Lisbon", False), ("type", "home.in", "2027-06-14", False), ("click", "home.flex"),
                    ("click", "home.search"), ("click", "stays.r2.deal"), ("click", "rooms.r1.reserve"),
                    ("type", "g.first", "Sam", False), ("type", "g.email", "sam@example.test", False), ("click", "g.confirm")]),
             ("/results", [("click", "res.breakfast"), ("click", "stays.r0.save"), ("click", "stays.r1.open"),
                           ("click", "rooms.r0.reserve"), ("click", "g.insure"), ("type", "g.last", "Lee", False),
                           ("click", "g.confirm")])]
    return Site("booking", "stay.example.test", pages, tasks)


TRAIN_SITES = [make_shop, make_admin]
HELD_OUT_SITES = [make_helpdesk, make_booking]


# ---------- drift: what changes between a recording and a replay ----------
def d_banner(page, rng):
    extra = [T("We use cookies to improve your experience."), C("banner.accept", "button", "Accept all", effect=("change",)),
             C("banner.reject", "button", rng.choice(["Reject", "Close", "Manage choices"]), effect=("change",))]
    page.items[0:0] = extra
    return True


def _pick_list(page, rng, at_least=1):
    lists = [l for l in page.factories if len(page.rows(l)[0]) >= at_least]
    return rng.choice(sorted(lists)) if lists else None


def d_reorder(page, rng):
    lst = _pick_list(page, rng, 2)
    if not lst:
        return False
    rows, start = page.rows(lst)
    before = [r for r, _ in rows]
    for _ in range(5):
        rng.shuffle(rows)
        if [r for r, _ in rows] != before:
            break
    page.set_rows(lst, rows, start)
    return True


def d_insert(page, rng):
    lst = _pick_list(page, rng)
    if not lst:
        return False
    rows, start = page.rows(lst)
    for _ in range(rng.randrange(1, 4)):
        page.fresh += 1
        rid = f"n{page.fresh}"
        try:
            rows.insert(rng.randrange(0, len(rows) + 1), (rid, page.factories[lst](rid)))
        except StopIteration:  # the generator ran out of names
            break
    page.set_rows(lst, rows, start)
    return True


def d_remove(page, rng):
    lst = _pick_list(page, rng, 2)
    if not lst:
        return False
    rows, start = page.rows(lst)
    rows.pop(rng.randrange(len(rows)))
    page.set_rows(lst, rows, start)
    return True


def d_drop(page, rng):
    """A control that is not in a list disappears."""
    solo = [it for it in page.controls() if it["list"] is None and not it["key"].startswith(("banner.", "decoy."))]
    if len(solo) < 2:
        return False
    page.items.remove(rng.choice(solo))
    return True


def d_rename(page, rng):
    names = sorted({it["name"] for it in page.controls() if it["name"] in SYNONYMS and it["name"] not in page.relabel})
    if not names:
        return False
    old = rng.choice(names)
    page.relabel[old] = SYNONYMS[old]
    return True


def d_role(page, rng):
    names = sorted({it["name"] for it in page.controls() if it["role"] in ("button", "link")} - page.rerole)
    if not names:
        return False
    old = rng.choice(names)
    page.rerole.add(old)
    return True


def d_numbers(page, rng):
    bump = lambda m: m.group(1) + str(int(m.group(2)) + rng.randrange(1, 40)) + m.group(3)
    pat = re.compile(r"(\$|\(|invited |by \w+ )(\d+)(\)| days| hours| per night|$|\b)")
    hit = False
    for it in page.items:
        field = "name" if it["k"] == "c" else "text"
        new = pat.sub(bump, it[field])
        hit, it[field] = hit or new != it[field], new
    return hit


def d_heading(page, rng):
    heads = [it for it in page.items if it["k"] == "h"]
    if not heads:
        return False
    h = rng.choice(heads)
    h["text"] = HEADINGS.get(h["text"], h["text"] + f" ({rng.randrange(2, 30)})")
    return True


def d_noise(page, rng):
    lst = _pick_list(page, rng)
    page.fresh += 1
    if lst and rng.random() < 0.7:
        rows, start = page.rows(lst)
        role, name = rng.choice([("link", "Details"), ("button", "Share"), ("button", "Compare"), ("link", "History")])
        at = rng.randrange(1, 4)
        for rid, items in rows:
            items.insert(min(at, len(items)), C(f"noise{page.fresh}.{rid}", role, name, effect=("change",), row=rid, lst=lst))
        page.set_rows(lst, rows, start)
    else:
        at = rng.randrange(0, len(page.items) + 1)
        row = page.items[at - 1].get("row") if at and at < len(page.items) and page.items[at - 1].get("row") == page.items[at].get("row") else None
        lst = page.items[at - 1].get("list") if row else None
        page.items.insert(at, T(rng.choice(["Free shipping this week.", "New: bulk actions.", "Updated 5 minutes ago.",
                                            "Limited availability."]), row, lst))
    return True


def d_decoy(page, rng):
    """A second control with the role and name of one that used to be unique, somewhere else on the page."""
    counts = {}
    for it in page.controls():
        counts[(it["role"], it["name"])] = counts.get((it["role"], it["name"]), 0) + 1
    solo = [it for it in page.controls() if counts[(it["role"], it["name"])] == 1 and it["role"] in ("button", "link", "textbox", "searchbox")]
    if not solo:
        return False
    src = rng.choice(solo)
    page.fresh += 1
    extra = [H(rng.choice(["Quick links", "More", "Shortcuts"])), T("Popular right now"),
             C(f"decoy.{page.fresh}", src["role"], src["name"], effect=("change",))]
    if rng.random() < 0.5:
        page.items.extend(extra)
    else:
        page.items[0:0] = extra
    return True


def d_replace(page, rng):
    """A feature is taken out and a different one put in its place. The old control is gone."""
    pool = [it for it in page.controls() if it["role"] in ("button", "link") and not it["key"].startswith(("banner.", "decoy.", "noise", "repl"))]
    if not pool:
        return False
    src = rng.choice(pool)
    page.fresh += 1
    name = rng.choice(["Compare", "Share", "Gift wrap", "Notify me", "Archive", "Duplicate", "Export", "Print"])
    for it in page.controls():
        if it["list"] == src["list"] and it["role"] == src["role"] and it["name"] == src["name"] and (src["list"] or it is src):
            it.update(key=f"repl{page.fresh}.{it['row']}", name=name, effect=("change",))
    return True


def d_toggle(page, rng):
    toggles = [it for it in page.controls() if it["effect"] == ("toggle",)]
    if not toggles:
        return False
    _flip(rng.choice(toggles))
    return True


DRIFTS = {"banner": d_banner, "reorder": d_reorder, "insert": d_insert, "remove": d_remove, "drop": d_drop,
          "rename": d_rename, "role": d_role, "numbers": d_numbers, "heading": d_heading, "noise": d_noise,
          "decoy": d_decoy, "toggle": d_toggle, "replace": d_replace}
# Relative frequencies. The held-out mix leans on the harder changes (removals, decoys, relabels).
TRAIN_MIX = {"banner": 2, "reorder": 3, "insert": 3, "remove": 2, "drop": 1, "rename": 2, "role": 1, "numbers": 2,
             "heading": 1, "noise": 2, "decoy": 1, "toggle": 1, "replace": 1}
HELD_OUT_MIX = {"banner": 1, "reorder": 3, "insert": 2, "remove": 3, "drop": 2, "rename": 3, "role": 2, "numbers": 1,
                "heading": 2, "noise": 3, "decoy": 3, "toggle": 1, "replace": 2}


def drift_page(page, rng, k, mix):
    """A changed copy of `page` with k drift operations applied, and their names."""
    page = copy.deepcopy(page)
    names, weights = zip(*sorted(mix.items()))
    done = []
    for _ in range(k * 4):
        if len(done) == k:
            break
        name = rng.choices(names, weights)[0]
        if name not in done and DRIFTS[name](page, rng):
            done.append(name)
    for it in page.controls():  # a relabel reaches every control with that name, new rows included
        if it["name"] in page.rerole and it["role"] in ("button", "link") and not it["key"].startswith("noise"):
            it["role"] = "link" if it["role"] == "button" else "button"
        it["name"] = page.relabel.get(it["name"], it["name"])
    page.relabel, page.rerole = {}, set()
    return page, done


def drift_site(site, rng, k, mix):
    out = copy.copy(site)
    out.pages = {}
    applied = {}
    for path, page in site.pages.items():
        out.pages[path], applied[path] = drift_page(page, rng, k, mix)
    return out, applied


# ---------- the stand-in browser ----------
def _flip(it):
    it["states"] = ["checked" if s == "unchecked" else "unchecked" if s == "checked" else s for s in it["states"]]


class Browser:
    """One tab on a synthetic site. `hits` lists the key of every element an action reached."""

    def __init__(self, site, rng):
        self.site, self.rng = site, rng
        self.next_id = rng.randrange(20, 4000)
        self.path, self.page, self.ids, self.hits, self.loads = None, None, {}, [], 0

    def goto_url(self, url):
        path = "/" + url.split("://", 1)[-1].partition("/")[2].split("?")[0]
        self.path, self.loads = path, self.loads + 1
        self.page = copy.deepcopy(self.site.pages.get(path) or Page("Not found", [H("404"), T("This page does not exist.")]))
        self.ids = {}
        for it in self.page.controls():  # a new document: every backendNodeId is new
            self.next_id += self.rng.randrange(1, 9)
            self.ids[self.next_id] = it

    new_tab = goto_url

    @property
    def url(self):
        return self.site.url(self.path)

    def id_of(self, key):
        return next((i for i, it in self.ids.items() if it["key"] == key), None)


class FakeBu:
    """state / click / type with the signatures and printed text of browser_use.ax.AxHelpers."""

    def __init__(self, browser, log=None):
        self.b, self.out, self.last, self.acted = browser, [] if log is None else log, {}, False

    def state(self, max=3000):
        b, lines, seen = self.b, [], {}
        for it in b.page.items:
            if it["k"] == "h":
                lines.append("## " + it["text"])
            elif it["k"] == "t":
                lines.append(it["text"])
            else:
                id = next(i for i, x in b.ids.items() if x is it)
                e = Entry(id, it["role"], it["name"], it["value"], it["states"], it["options"],
                          new=bool(self.last) and id not in self.last)
                seen[id] = e
                lines.append(format_control(e))
        self.last, self.acted = seen, False
        text = format_state(b.page.title, b.url, lines)
        self.out.append(text)
        return text

    def _describe(self, id):
        return self.last[id].describe() if id in self.last else f"[{id}]"

    def _act(self, op, id, body):
        self.acted = True
        b = self.b
        it = b.ids.get(id)
        if it is None:
            raise RuntimeError(f"{op} {self._describe(id)} failed: CDP -32000: No node with given id found")
        if "disabled" in it["states"]:
            raise RuntimeError(f"{op} {self._describe(id)} failed: Error: Target disabled")
        described, loads = self._describe(id), b.loads
        detail, effect = body(it)
        b.hits.append(it["key"])
        if effect and effect[0] == "nav":
            b.goto_url(b.site.url(effect[1]))
            outcome = f"navigated to {b.url}"
        elif effect and effect[0] == "toggle":
            _flip(it)
            outcome = "page changed"
        else:
            outcome = "page changed" if effect else "no change"
        assert loads == b.loads or outcome.startswith("navigated")
        line = f"[ok] {op} {described}{detail} -> {outcome}"
        self.out.append(line)
        return line

    def click(self, id):
        return self._act("click", id, lambda it: ("", it["effect"]))

    def type(self, id, text, enter=False):
        if not isinstance(text, str):
            raise TypeError("type needs a string.")

        def body(it):
            if it["role"] == "select":
                label = next((o for o in it["options"] if o.lower() == text.lower()), None)
                if label is None:
                    raise RuntimeError(f"type {self._describe(id)} failed: Error: no option {quote(text)}")
                it["value"] = label
                return f" = {quote(label)}", ("change",)
            it["value"] = text
            shown = f"<{len(text)} characters, hidden>" if it["password"] else quote(clip(text, DESCRIBE_CLIP))
            return f" = {shown}" + (" +Enter" if enter else ""), (it["enter"] if enter and it["enter"] else ("change",))

        return self._act("type", id, body)


# ---------- recording a scripted run as a transcript ----------
def record(site, task, rng, chain=0.3):
    """Run `task` on `site` and return (messages, keys): the transcript and the element each action hit."""
    start, intents = task
    browser = Browser(site, rng)
    messages, n = [], 0

    def cell(code, run):
        nonlocal n
        n += 1
        out = []
        bu.out = out
        error = None
        try:
            run()
        except RuntimeError as e:
            error = f"RuntimeError: {e}"
        if bu.acted:  # the worker prints the page after a cell whose actions reached it
            bu.state()
        text = "\n".join(out) + (f"\nError: {error}" if error else "")
        messages.append({"kind": "response", "parts": [{"part_kind": "tool-call", "tool_name": "python",
                                                        "tool_call_id": f"call_{n}", "args": {"code": code}}]})
        messages.append({"kind": "request", "parts": [{"part_kind": "tool-return", "tool_name": "python",
                                                       "tool_call_id": f"call_{n}", "content": [text]}]})

    bu = FakeBu(browser)
    messages.append({"kind": "request", "parts": [{"part_kind": "user-prompt", "content": f"SYNTHETIC task on {site.name}"}]})
    url = site.url(start)
    cell(f"new_tab({url!r})\nbu.state()", lambda: (browser.new_tab(url), bu.state()))
    i = 0
    while i < len(intents):
        batch = [intents[i]]
        nav = lambda it: (it["effect"] or ("",))[0] == "nav" or (len(batch[-1]) > 3 and batch[-1][3] and it["enter"])
        # Chain a second action into the same cell only when the first one stays on the page,
        # as a model would: after a navigation it has no ids for the new page yet.
        if i + 1 < len(intents) and rng.random() < chain and not nav(browser.ids[browser.id_of(batch[0][1])]):
            batch.append(intents[i + 1])
        calls, code = [], []
        for intent in batch:
            id = browser.id_of(intent[1])
            if id is None:
                raise KeyError(f"task refers to {intent[1]!r}, which is not on {browser.path}")
            if intent[0] == "click":
                code.append(f"bu.click({id})")
                calls.append(lambda id=id: bu.click(id))
            else:
                code.append(f"bu.type({id}, {intent[2]!r}" + (", enter=True)" if intent[3] else ")"))
                calls.append(lambda id=id, intent=intent: bu.type(id, intent[2], enter=intent[3]))
        cell("\n".join(code), lambda: [c() for c in calls])
        i += len(batch)
    messages.append({"kind": "response", "parts": [{"part_kind": "tool-call", "tool_name": "finish",
                                                    "tool_call_id": "call_finish", "args": {"result": "done", "success": True}}]})
    return messages, list(browser.hits)


def dumps(messages):
    return json.dumps({"version": 1, "summary": "SYNTHETIC transcript written by sim.py", "covered_messages": len(messages),
                       "messages": messages}, indent=1, ensure_ascii=False)
