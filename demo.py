"""A one-minute walk through refind on SYNTHETIC data (sim.py). No browser, no network, no model.

    python demo.py
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim
from replay import replay, handoff_prompt
from transcript import compile_plan


def show(title):
    print(f"\n=== {title} ===")


def run(plan, site, seed, keys):
    browser = sim.Browser(site, random.Random(seed))
    result = replay(plan, sim.FakeBu(browser), goto=browser.goto_url)
    for row in result["log"]:
        if row["kind"] == "goto":
            print(f"  goto   {row['url']}")
        else:
            print(f"  {row['kind']:<6} new id {row['id']:<5} {row['target']:<44} score {row['confidence']:.2f} -> {row['outcome']}")
    right = sum(a == b for a, b in zip(browser.hits, keys))
    print(f"  status: {result['status']}; {len(browser.hits)} actions sent, {right} on the recorded element (ground truth from the simulator)")
    return result


def main():
    rng = random.Random(7)
    site = sim.make_admin(rng)
    task = site.tasks[1]

    show("1. A recorded run (synthetic transcript in the shape of browser-use 1.0 result.messages)")
    messages, keys = sim.record(site, task, rng, chain=0)
    cells = [p for m in messages for p in m["parts"] if p["part_kind"] in ("tool-call", "tool-return")]
    print("first cell :", cells[0]["args"]["code"].replace("\n", "; "))
    print("its output :", "\n             ".join(cells[1]["content"][0].split("\n")[:8]), "\n             ...")
    print("next cell  :", cells[2]["args"]["code"])
    print("its output :", cells[3]["content"][0].split("\n")[0])

    show("2. Compile: every literal id becomes a fingerprint")
    plan = compile_plan(json.loads(sim.dumps(messages)))
    print("counts :", plan["counts"])
    step = next(s for s in plan["steps"] if s["kind"] == "click")
    fp = {k: v for k, v in step["fp"].items() if k != "known"}
    print("step   :", step["kind"], step["target"], "expect", step["expect"])
    print("fp     :", json.dumps(fp))

    show("3. Replay on the same site in a new session (every id is different)")
    run(plan, site, 11, keys)

    show("4. Replay after the site changed: banner, rows reordered and added, a relabel")
    changed = sim.Site(site.name, site.host, dict(site.pages), site.tasks)
    page = sim.copy.deepcopy(site.pages["/team"])
    for op in ("banner", "reorder", "insert", "numbers"):
        sim.DRIFTS[op](page, rng)
    changed.pages["/team"] = page
    edit = sim.copy.deepcopy(site.pages["/edit"])
    next(c for c in edit.controls() if c["key"] == "edit.save")["name"] = "Save"
    changed.pages["/edit"] = edit
    run(plan, changed, 12, keys)

    show("5. Replay after the member the run removes at the end was already taken off the page")
    gone = sim.Site(site.name, site.host, dict(changed.pages), site.tasks)
    page = sim.copy.deepcopy(changed.pages["/team"])
    page.items = [it for it in page.items if not (it.get("list") == "members" and it.get("row") == "r1")]
    gone.pages["/team"] = page
    result = run(plan, gone, 13, keys)
    print("  handoff:", json.dumps(result["handoff"]))
    print("\n" + handoff_prompt(plan, result))


if __name__ == "__main__":
    main()
