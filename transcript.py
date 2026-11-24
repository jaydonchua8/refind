"""Turn a recorded browser-use 1.0 run into a replay plan.

Input: the run's messages as JSON. browser-use 1.0 has no per-step history any more
(AgentHistoryList.history, save_to_file and Agent.rerun_history raise a "removed" error); what
a run leaves is `result.messages`, a Pydantic AI transcript. The JSON shape read here is the one
browser_use/agent/loop.py `_plain()` writes into its compaction archive:

    [{"kind": "response", "parts": [{"part_kind": "tool-call", "tool_name": "python",
                                     "tool_call_id": "c1", "args": {"code": "bu.click(1400)"}}]},
     {"kind": "request",  "parts": [{"part_kind": "tool-return", "tool_name": "python",
                                     "tool_call_id": "c1", "content": ["[ok] click [1400] ..."]}]}]

either as a bare list or under a "messages" key (the archive file is
{"version": 1, "summary": ..., "covered_messages": ..., "messages": [...]}).

Only ultrafast-mode runs can be compiled, because only there does the output contain the
`bu.state()` listing that says what an id was. A cell is replayed only when every statement in it
is understood. Anything else becomes an "opaque" step, where the replay stops and hands back.
"""
import ast
import json

from axstate import split_output, parse_type_detail, clip, DESCRIBE_CLIP
from refind import fingerprint, VERSION

DELIVERY_TOOLS = {"finish", "finish_from_repl"}
NAVIGATE = {"new_tab", "goto_url"}
IGNORED_CALLS = {"wait_for_load", "wait_for_network_idle"}


class TranscriptError(ValueError):
    pass


def read_cells(data):
    """[(tool_name, args dict, output text or None)] in call order."""
    messages = data.get("messages") if isinstance(data, dict) else data
    if not isinstance(messages, list):
        raise TranscriptError("expected a list of messages or an object with a 'messages' list")
    calls, returns = [], {}
    for m in messages:
        if not isinstance(m, dict):
            raise TranscriptError("a message is not an object")
        for p in m.get("parts") or []:
            kind = p.get("part_kind")
            if kind == "tool-call":
                args = p.get("args")
                if isinstance(args, str):  # Pydantic AI keeps args as a JSON string for some providers
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {}
                calls.append((p.get("tool_name"), args if isinstance(args, dict) else {}, p.get("tool_call_id")))
            elif kind == "tool-return":
                content = p.get("content")
                if isinstance(content, list):  # python returns [text, *screenshots]
                    content = "\n".join(c for c in content if isinstance(c, str))
                returns[p.get("tool_call_id")] = content if isinstance(content, str) else None
    return [(name, args, returns.get(call_id)) for name, args, call_id in calls]


def _literal(node, types):
    return isinstance(node, ast.Constant) and type(node.value) in types


def _call_of(stmt):
    """The single call a statement makes: `f(...)`, `x = f(...)` or `print(f(...))`."""
    if isinstance(stmt, ast.Expr):
        node = stmt.value
    elif isinstance(stmt, ast.Assign) and all(isinstance(t, ast.Name) for t in stmt.targets):
        node = stmt.value
    else:
        return None
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"
            and len(node.args) == 1 and not node.keywords):
        node = node.args[0]
    return node if isinstance(node, ast.Call) else None


def read_code(code):
    """The cell as a list of understood operations, or (None, reason).

    Understood: new_tab('url'), goto_url('url'), wait_for_load(), bu.state(), bu.click(<int>),
    bu.type(<int>, '<text>', enter=<bool>), each as its own statement with literal arguments.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return None, f"cell does not parse: {e.msg}"
    ops = []
    for stmt in tree.body:
        call = _call_of(stmt)
        if call is None:
            return None, f"line {stmt.lineno}: not a plain helper call"
        f, kw = call.func, {k.arg: k.value for k in call.keywords}
        if isinstance(f, ast.Name) and f.id in NAVIGATE and len(call.args) == 1 and not kw and _literal(call.args[0], (str,)):
            ops.append({"kind": "goto", "via": f.id, "url": call.args[0].value})
        elif isinstance(f, ast.Name) and f.id in IGNORED_CALLS and not call.args:
            continue
        elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "bu":
            if f.attr == "state" and not call.args and set(kw) <= {"max"}:
                ops.append({"kind": "look"})
            elif f.attr == "click" and len(call.args) == 1 and not kw and _literal(call.args[0], (int,)):
                ops.append({"kind": "click", "id": call.args[0].value})
            elif (f.attr == "type" and len(call.args) in (2, 3) and set(kw) <= {"enter"}
                  and _literal(call.args[0], (int,)) and _literal(call.args[1], (str,))
                  and all(_literal(v, (bool,)) for v in [*call.args[2:], *kw.values()])
                  and not (len(call.args) == 3 and kw)):
                enter = call.args[2].value if len(call.args) == 3 else (kw["enter"].value if kw else False)
                ops.append({"kind": "type", "id": call.args[0].value, "text": call.args[1].value, "enter": enter})
            else:
                return None, f"line {stmt.lineno}: bu.{f.attr} call with arguments that are not literals"
        else:
            return None, f"line {stmt.lineno}: {ast.unparse(f)}() is not replayed"
    return ops, ""


def has_node_ids(code):
    """True when the code could hold a session-specific id (any integer literal of 10 or more)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True
    return any(_literal(n, (int,)) and n.value >= 10 for n in ast.walk(tree))


def compile_plan(data):
    """A replay plan: goto / click / type steps with fingerprints, and opaque steps where it must stop."""
    steps, state, stale = [], None, False
    counts = {"cells": 0, "plain_cells": 0, "actions": 0, "opaque": 0}

    def opaque(i, code, reason):
        counts["opaque"] += 1
        steps.append({"kind": "opaque", "cell": i, "reason": reason, "code": code, "id_free": not has_node_ids(code)})

    for i, (tool, args, output) in enumerate(read_cells(data)):
        if tool in DELIVERY_TOOLS:
            steps.append({"kind": "deliver", "cell": i, "tool": tool})
            break
        if tool != "python":
            opaque(i, json.dumps(args), f"tool {tool!r} is not replayed")
            continue
        counts["cells"] += 1
        code = args.get("code")
        if not isinstance(code, str) or output is None:
            opaque(i, code if isinstance(code, str) else "", "the call has no code or no recorded output")
            state = None
            continue
        events = list(split_output(output))
        ops, why = read_code(code)
        problem = why
        cell_steps = []
        if ops is not None:
            problem = _align(ops, events, state, stale, cell_steps, i)
        # Whatever happened, later cells resolve ids against the last listing this cell printed.
        for kind, value in events:
            if kind == "state":
                state, stale = value, False
            elif kind == "action" and value["outcome"] == "navigated":
                stale = True
            elif kind == "state_error":
                state = None
        if ops is not None and any(o["kind"] == "goto" for o in ops) and not any(k == "state" for k, _ in events):
            stale = True
        if problem:
            opaque(i, code, problem)
        else:
            counts["plain_cells"] += 1
            counts["actions"] += sum(s["kind"] in ("click", "type") for s in cell_steps)
            steps.extend(cell_steps)
    return {"v": VERSION, "steps": steps, "counts": counts}


def _align(ops, events, state, stale, out, cell):
    """Pair each code operation with the line it printed. Returns '' or the reason the cell is opaque."""
    if any(k == "failed" for k, _ in events):
        return "an action failed in the recorded run, so what the page did is not known"
    if any(k == "state_error" for k, _ in events):
        return "a page listing in the output is incomplete"
    stream = [(k, v) for k, v in events if k in ("state", "action")]
    pos = 0
    for op in ops:
        if op["kind"] == "goto":
            if "<secret:" in op["url"]:
                return "the url holds a redacted secret"
            out.append({"kind": "goto", "cell": cell, "url": op["url"], "via": op["via"]})
            stale = True
            continue
        if op["kind"] == "look":
            if pos >= len(stream) or stream[pos][0] != "state":
                return "bu.state() printed no listing"
            state, stale, pos = stream[pos][1], False, pos + 1
            continue
        if pos >= len(stream) or stream[pos][0] != "action":
            return "the output has no [ok] line for an action in the code"
        act, pos = stream[pos][1], pos + 1
        if act["op"] != op["kind"] or act["id"] != op["id"]:
            return "the [ok] lines do not match the code"
        if state is None:
            return f"id {op['id']} was used before any page listing"
        if stale:
            return f"id {op['id']} was used after a navigation with no new listing"
        entry = state.by_id(op["id"])
        if entry is None:
            return f"id {op['id']} is not in the last page listing"
        described = entry.describe()[len(f"[{entry.id}]"):]
        if not act["rest"].startswith(described):
            return "the [ok] line names a different element than the listing"
        step = {"kind": op["kind"], "cell": cell, "page": {"url": state.url, "title": state.title},
                "target": f'{entry.role} "{clip(entry.name, DESCRIBE_CLIP)}"',
                "fp": fingerprint(state, op["id"]),
                "expect": {"outcome": act["outcome"], "url": act["url"]}}
        if op["kind"] == "type":
            if "<secret:" in op["text"]:
                return "the typed text is a redacted secret"
            detail = parse_type_detail(act["rest"][len(described):])
            if detail is None:
                return "the type line could not be read"
            if detail["text"] is not None and detail["complete"] and detail["text"] != op["text"]:
                return "the printed text differs from the text in the code"
            step.update(text=op["text"], enter=op["enter"])
        out.append(step)
        if act["outcome"] == "navigated":
            stale = True
    if pos < len(stream) and stream[pos][0] == "state":
        pos += 1  # the listing the worker prints after a cell that acted
    if pos != len(stream):
        return "the output has more actions than the code"
    return ""
