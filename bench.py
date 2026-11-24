"""Benchmark on SYNTHETIC sites (sim.py): does a recorded action still land on the same element?

Two levels, each on a TRAIN split (the sites and drift mix the thresholds were tuned on) and a
HELD-OUT split (two other sites, another drift mix, other seeds, never used for tuning):

  elements  one control, recorded on a page, looked up on a drifted copy of that page
  tasks     a recorded multi-page run compiled from its transcript and replayed on a drifted site

    python bench.py            both splits, writes bench_results.json
    python bench.py --quick    fewer seeds (a few seconds)
    python bench.py --tune     random search for PARAMS on the train split only
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim
from axstate import parse_state, fold
from refind import fingerprint, components, decide, find, Match, PARAMS
from replay import replay
from transcript import compile_plan

SPLITS = {"train": (sim.TRAIN_SITES, sim.TRAIN_MIX, 0), "held_out": (sim.HELD_OUT_SITES, sim.HELD_OUT_MIX, 1000)}
KS = (1, 2, 3)


# ---------- baselines: what you get from the transcript without a fingerprint ----------
def _same(fp, state):
    return [e for e in state.entries if e.role == fp["role"] and fold(e.name) == fold(fp["name"])]


def raw_id(fp, state, params=None, id=None):
    e = state.by_id(id)
    return Match(e, "found" if e else "not_found")


def name_first(fp, state, params=None):
    hits = _same(fp, state)
    return Match(hits[0] if hits else None, "found" if hits else "not_found")


def name_nth(fp, state, params=None):
    hits = _same(fp, state)
    ok = fp["ordinal"] < len(hits)
    return Match(hits[fp["ordinal"]] if ok else None, "found" if ok else "not_found")


def name_unique(fp, state, params=None):
    hits = _same(fp, state)
    return Match(hits[0] if len(hits) == 1 else None, "found" if len(hits) == 1 else "ambiguous")


METHODS = {"recorded id": None, "role+name, first": name_first, "role+name, nth": name_nth,
           "role+name, only if unique": name_unique, "refind": find}


# ---------- element level ----------
def _look(site, path, rng):
    b = sim.Browser(site, rng)
    b.goto_url(site.url(path))
    state = parse_state(sim.FakeBu(b).state())
    return state, {id: it["key"] for id, it in b.ids.items()}


def element_cases(split, seeds, trials=2):
    """Yield (fingerprint, old id, drifted State, true new id or None, k)."""
    makers, mix, base = SPLITS[split]
    for seed in seeds:
        for make in makers:
            rng = random.Random(f"{make.__name__}/{base + seed}")
            site = make(rng)
            for path, page in site.pages.items():
                if not page.controls():
                    continue
                old, old_keys = _look(site, path, rng)
                prints = {id: fingerprint(old, id) for id in old_keys}
                for k in KS:
                    for _ in range(trials):
                        changed, _ops = sim.drift_page(page, rng, k, mix)
                        site2 = sim.Site(site.name, site.host, {**site.pages, path: changed}, site.tasks)
                        new, new_keys = _look(site2, path, rng)
                        where = {key: id for id, key in new_keys.items()}
                        for id, key in old_keys.items():
                            yield prints[id], id, new, where.get(key), k


def tally():
    return {"present": 0, "correct": 0, "wrong": 0, "refused": 0, "absent": 0, "absent_refused": 0, "absent_wrong": 0}


def score_into(c, got, truth):
    if truth is None:
        c["absent"] += 1
        c["absent_refused" if got is None else "absent_wrong"] += 1
    else:
        c["present"] += 1
        c["correct" if got == truth else "refused" if got is None else "wrong"] += 1


def run_elements(split, seeds):
    out = {name: {k: tally() for k in (*KS, "all")} for name in METHODS}
    for fp, old_id, new, truth, k in element_cases(split, seeds):
        for name, method in METHODS.items():
            m = raw_id(fp, new, id=old_id) if method is None else method(fp, new, PARAMS)
            for bucket in (k, "all"):
                score_into(out[name][bucket], m.id, truth)
    return out


# ---------- task level ----------
def oracle_feasible(site, task, keys, rng):
    """Could the recorded run be repeated on this site at all? Replays by ground-truth key."""
    b = sim.Browser(site, rng)
    bu = sim.FakeBu(b)
    b.goto_url(site.url(task[0]))
    for intent in task[1]:
        id = b.id_of(intent[1])
        if id is None:
            return False
        it = b.ids[id]
        if "disabled" in it["states"]:
            return False
        try:
            bu.click(id) if intent[0] == "click" else bu.type(id, intent[2], enter=intent[3])
        except RuntimeError:
            return False
    return True


def run_tasks(split, seeds, methods=("role+name, first", "role+name, nth", "role+name, only if unique", "refind")):
    makers, mix, base = SPLITS[split]
    out = {name: {"runs": 0, "completed_right": 0, "stopped_clean": 0, "wrong_action": 0, "wrong_then_completed": 0,
                  "actions": 0, "actions_replayed_right": 0, "feasible": 0, "feasible_completed": 0,
                  "infeasible": 0, "infeasible_wrong": 0, "stop_reasons": {}} for name in methods}
    for seed in seeds:
        for make in makers:
            rng = random.Random(f"task/{make.__name__}/{base + seed}")
            site = make(rng)
            for task in site.tasks:
                messages, keys = sim.record(site, task, rng)
                plan = compile_plan(json.loads(sim.dumps(messages)))
                n_actions = sum(s["kind"] in ("click", "type") for s in plan["steps"])
                assert n_actions == len(keys), "the compiler lost an action of a fully plain synthetic run"
                for k in KS:
                    drifted, _ = sim.drift_site(site, rng, k, mix)
                    # A toggle the recording flipped may start flipped on the drifted site; then the run
                    # cannot be repeated as recorded. Removed targets make it infeasible too.
                    feasible = oracle_feasible(drifted, task, keys, random.Random(1)) and _toggles_match(site, drifted, task)
                    for name in methods:
                        b = sim.Browser(drifted, random.Random(f"replay/{seed}/{k}"))
                        r = replay(plan, sim.FakeBu(b), goto=b.goto_url, finder=METHODS[name])
                        c = out[name]
                        right = sum(1 for a, w in zip(b.hits, keys) if a == w)
                        wrong = len(b.hits) - right
                        c["runs"] += 1
                        c["actions"] += len(keys)
                        c["actions_replayed_right"] += right if not wrong else 0
                        c["feasible" if feasible else "infeasible"] += 1
                        if wrong:
                            c["wrong_action"] += 1
                            c["wrong_then_completed"] += r["status"] == "completed"
                            c["infeasible_wrong"] += not feasible
                        elif r["status"] == "completed":
                            c["completed_right"] += 1
                            c["feasible_completed"] += feasible
                        else:
                            c["stopped_clean"] += 1
                            why = r["handoff"]["reason"]
                            c["stop_reasons"][why] = c["stop_reasons"].get(why, 0) + 1
    return out


def _toggles_match(site, drifted, task):
    for intent in task[1]:
        if intent[0] != "click":
            continue
        for path, page in site.pages.items():
            a = next((it for it in page.controls() if it["key"] == intent[1]), None)
            b = next((it for it in drifted.pages[path].controls() if it["key"] == intent[1]), None)
            if a and b and a["effect"] == ("toggle",) and a["states"] != b["states"]:
                return False
    return True


# ---------- tuning (train split only) ----------
def tune(seeds, samples=400):
    cases = [(fp, components(fp, new), truth) for fp, _, new, truth, _ in element_cases("train", seeds) if not new.truncated]
    rng = random.Random(2027)

    def value(p):
        c = tally()
        for fp, comps, truth in cases:
            score_into(c, decide(comps, fp, p).id, truth)
        # A wrong click costs far more than a refusal, which only costs one model call.
        return c["correct"] + c["absent_refused"] - 25 * (c["wrong"] + c["absent_wrong"]), c

    best = (value(PARAMS)[0], dict(PARAMS))
    print(f"{len(cases)} train cases; current PARAMS score {best[0]}")
    for _ in range(samples):
        p = {"w_name": rng.choice([1.5, 2, 3, 4]), "w_ctx": rng.choice([1.5, 2, 3, 4, 5]), "w_head": rng.choice([0, 0.5, 1, 1.5]),
             "w_ord": rng.choice([0, 0.25, 0.5, 1]), "threshold": rng.choice([0.5, 0.55, 0.6, 0.65, 0.7]),
             "margin": rng.choice([0.03, 0.05, 0.08, 0.12]), "ctx_min": rng.choice([0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
             "rename_ctx": rng.choice([0.6, 0.7, 0.8, 0.9, 0.95]), "solo_ctx": rng.choice([0.0, 0.1, 0.2, 0.3]),
             "rename_name": rng.choice([0.2, 0.25, 0.35, 0.5, 2.0])}
        v = value(p)[0]
        if v > best[0]:
            best = (v, p)
            print(v, json.dumps(p))
    print("best:", json.dumps(best[1]), value(best[1])[1])


# ---------- report ----------
def pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "   n/a"


def print_elements(title, res):
    print(f"\n{title}")
    print(f"{'method':<28}{'still there':>12}{'found':>8}{'wrong':>8}{'refused':>9}{'removed':>9}{'refused':>9}{'wrong':>8}")
    for name, by_k in res.items():
        c = by_k["all"]
        print(f"{name:<28}{c['present']:>12}{pct(c['correct'], c['present']):>8}{pct(c['wrong'], c['present']):>8}"
              f"{pct(c['refused'], c['present']):>9}{c['absent']:>9}{pct(c['absent_refused'], c['absent']):>9}"
              f"{pct(c['absent_wrong'], c['absent']):>8}")


def print_tasks(title, res):
    print(f"\n{title}")
    print(f"{'method':<28}{'runs':>6}{'completed':>11}{'stopped':>9}{'wrong click':>13}{'feasible done':>15}{'actions w/o model':>19}")
    for name, c in res.items():
        print(f"{name:<28}{c['runs']:>6}{pct(c['completed_right'], c['runs']):>11}{pct(c['stopped_clean'], c['runs']):>9}"
              f"{pct(c['wrong_action'], c['runs']):>13}{pct(c['feasible_completed'], c['feasible']):>15}"
              f"{pct(c['actions_replayed_right'], c['actions']):>19}")


def main(argv):
    quick = "--quick" in argv
    if "--tune" in argv:
        return tune(range(8 if quick else 16))
    n = 6 if quick else 20
    results = {"note": "SYNTHETIC sites from sim.py. held_out sites, drift mix and seeds were not used for tuning.",
               "params": PARAMS, "seeds_per_split": n}
    for split in SPLITS:
        e = run_elements(split, range(n))
        t = run_tasks(split, range(n))
        results[split] = {"elements": {m: {str(k): v for k, v in by.items()} for m, by in e.items()}, "tasks": t}
        print_elements(f"ELEMENTS, {split} (k = 1 to 3 changes per page)", e)
        print_tasks(f"TASKS, {split} (every page of the site drifted)", t)
        print("refind stop reasons:", json.dumps(t["refind"]["stop_reasons"], sort_keys=True))
    if not quick:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_results.json"), "w") as f:
            json.dump(results, f, indent=1, sort_keys=True)


if __name__ == "__main__":
    main(sys.argv[1:])
