"""Replay a compiled plan without a model, and stop the moment it is not sure.

`bu` is any object with the three ultrafast helpers of browser-use 1.0 (browser_use/ax.py,
class AxHelpers), with the same signatures and return text:

    bu.state()                      -> the "[state] ... [/state]" listing as a str
    bu.click(id)                    -> '[ok] click [id] role "name" -> <outcome>'
    bu.type(id, text, enter=False)  -> '[ok] type [id] role "name" = "text" -> <outcome>'

Both raise RuntimeError when the action fails. `goto` is a callable like the harness helper
goto_url(url). In this repository the only `bu` that exists is the simulator in sim.py: this
hook has not been run against the real library or a real browser.

Before each action the replay looks at the page again and checks the page address, re-finds the
element, and checks toggles are in the recorded state. After each action it compares the outcome
line with the recorded one. Any mismatch ends the replay with a handoff that says which step,
why, and whether the last action had already been sent.
"""
from urllib.parse import urlsplit

from axstate import parse_state, split_output, StateError
from refind import find, format_entry, PARAMS, TOGGLES


def same_page(a, b):
    """Same scheme, host and path. Query strings and fragments are ignored."""
    x, y = urlsplit(a), urlsplit(b)
    return (x.scheme, x.netloc, x.path.rstrip("/")) == (y.scheme, y.netloc, y.path.rstrip("/"))


def replay(plan, bu, goto=None, run_cell=None, params=PARAMS, finder=find):
    """Run plan["steps"] in order. Returns {"status": "completed" | "handoff", "done", "total", "log", "handoff"}."""
    steps = plan["steps"]
    log = []

    def stop(i, reason, detail="", acted=False):
        return {"status": "handoff", "done": i, "total": len(steps), "log": log,
                "handoff": {"step": i, "kind": steps[i]["kind"], "reason": reason, "detail": detail, "acted": acted}}

    for i, step in enumerate(steps):
        kind = step["kind"]
        if kind == "deliver":
            break
        if kind == "goto":
            if goto is None:
                return stop(i, "no_goto", "no goto callable was given")
            try:
                goto(step["url"])
            except Exception as e:
                return stop(i, "goto_failed", str(e), acted=True)
            log.append({"step": i, "kind": "goto", "url": step["url"]})
            continue
        if kind == "opaque":
            if run_cell is None or not step.get("id_free"):
                return stop(i, "opaque_cell", step["reason"])
            try:
                run_cell(step["code"])
            except Exception as e:
                return stop(i, "cell_failed", f"{type(e).__name__}: {e}", acted=True)
            log.append({"step": i, "kind": "opaque", "ran": True})
            continue
        try:
            state = parse_state(str(bu.state()))
        except StateError as e:
            return stop(i, "state_unreadable", str(e))
        if not same_page(state.url, step["page"]["url"]):
            return stop(i, "url_differs", f"on {state.url}, recorded on {step['page']['url']}")
        m = finder(step["fp"], state, params)
        if m.entry is None:
            return stop(i, m.status, f"best score {m.confidence:.2f}, {m.margin:.2f} ahead of the next candidate"
                        + (f" ({format_entry(m.runner_up)})" if m.runner_up else ""))
        e = m.entry
        if "disabled" in e.states:
            return stop(i, "disabled", f"{format_entry(e)} is disabled now")
        if kind == "click" and e.role in TOGGLES:
            was = [s for s in step["fp"]["states"] if s in ("checked", "unchecked")]
            now = [s for s in e.states if s in ("checked", "unchecked")]
            if was != now:
                return stop(i, "state_differs", f"{format_entry(e)} was {was or ['unset']} when recorded, is {now or ['unset']} now")
        try:
            line = bu.click(e.id) if kind == "click" else bu.type(e.id, step["text"], enter=step["enter"])
        except RuntimeError as err:
            return stop(i, "action_failed", str(err), acted=True)
        got = next((v for k, v in split_output(str(line)) if k == "action"), None)
        log.append({"step": i, "kind": kind, "id": e.id, "target": format_entry(e),
                    "confidence": round(m.confidence, 3), "margin": round(m.margin, 3),
                    "outcome": got["outcome"] if got else None})
        want = step["expect"]
        if got is None:
            return stop(i, "outcome_unreadable", str(line), acted=True)
        if got["outcome"] != want["outcome"] or (want["outcome"] == "navigated" and not same_page(got["url"], want["url"])):
            return stop(i, "outcome_differs",
                        f"recorded {want['outcome']} {want['url']}".strip() + f", got {got['outcome']} {got['url']}".rstrip(),
                        acted=True)
    return {"status": "completed", "done": len(steps), "total": len(steps), "log": log, "handoff": None}


def handoff_prompt(plan, result):
    """Text for continuing with a model after a handoff (for example through Agent.add_new_task)."""
    if result["status"] == "completed":
        return ""
    h = result["handoff"]
    step = plan["steps"][h["step"]]
    did = [f"- {x['kind']} {x.get('target') or x.get('url') or ''}".rstrip() for x in result["log"]]
    what = {"click": lambda: f"click {step['target']}",
            "type": lambda: f"type {step['text']!r} into {step['target']}" + (" and press Enter" if step["enter"] else ""),
            "goto": lambda: f"open {step['url']}",
            "opaque": lambda: "run this cell:\n" + step["code"]}[h["kind"]]()
    return "\n".join([
        "A recorded run was replayed without a model up to this point.",
        "Already done and checked:" if did else "Nothing was done yet.", *did,
        f"The replay stopped at the next step ({h['reason']}: {h['detail']}).",
        "That action was already sent and its result was not the recorded one; look at the page before doing anything."
        if h["acted"] else "That step was not attempted.",
        f"The recorded run went on to {what}. Look at the page and continue the task from here.",
    ])
