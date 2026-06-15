"""Identity-metadata tests for TrajectoryManager (origin / compact / caller).

Standalone -- does NOT import the e2e dump helpers (absent on this branch). Each
test drives ``record_turn`` / ``get_trajectory`` directly and asserts on the
``sample.metadata['identity']`` block:

    {origin, is_compact_start, node_list, start_node_id, caller_node_ids, match_kind}

Token ids are kept minimal; routing is driven by message-dict equality, and
linearization just needs each turn's prompt to prefix-extend the prior held
tokens (so segments stay CLEAN unless a test deliberately drifts).
"""

from __future__ import annotations

from slime.agent.trajectory import COMPACT_SUMMARY_PREFIX, TrajectoryManager, TurnRecord
from slime.utils.types import Sample


def _base() -> Sample:
    return Sample(index=0, group_index=0, prompt="p", label="l")


def _agent_call(prompt: str) -> dict:
    """An assistant message issuing an Agent tool call with ``prompt``."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": "Agent", "arguments": {"prompt": prompt}}}],
    }


def _rec(mgr, sid, prompt_messages, response_message, *, prompt_ids, output_ids):
    mgr.record_turn(
        sid,
        turn=TurnRecord(prompt_ids=list(prompt_ids), output_ids=list(output_ids), finish_reason="stop"),
        prompt_messages=prompt_messages,
        response_message=response_message,
    )


def _identities(samples):
    return [s.metadata["identity"] for s in samples]


# ---------------------------------------------------------------------------
# main-only
# ---------------------------------------------------------------------------


def test_main_only_single_segment():
    mgr = TrajectoryManager()
    sid = "main"
    sysm = {"role": "system", "content": "MAIN-SYS"}
    _rec(
        mgr,
        sid,
        [sysm, {"role": "user", "content": "do x"}],
        {"role": "assistant", "content": "a1"},
        prompt_ids=[1, 2, 3],
        output_ids=[10],
    )
    _rec(
        mgr,
        sid,
        [
            sysm,
            {"role": "user", "content": "do x"},
            {"role": "assistant", "content": "a1"},
            {"role": "tool", "content": "r"},
        ],
        {"role": "assistant", "content": "a2"},
        prompt_ids=[1, 2, 3, 10, 4],
        output_ids=[11],
    )
    samples = mgr.get_trajectory(sid, base_sample=_base())
    assert len(samples) == 1
    idn = samples[0].metadata["identity"]
    assert idn["origin"] == "main"
    assert idn["is_compact_start"] is False
    assert idn["caller_node_ids"] == []
    assert idn["match_kind"] is None
    assert idn["node_list"] == idn["node_list"]  # present
    assert idn["start_node_id"] == idn["node_list"][0]
    assert len(idn["node_list"]) == 2  # two generated turns in one segment


# ---------------------------------------------------------------------------
# sub-agent (exact prompt match) + caller linkage
# ---------------------------------------------------------------------------


def test_sub_agent_exact_match_and_caller():
    mgr = TrajectoryManager()
    sid = "sub"
    main_sys = {"role": "system", "content": "MAIN-SYS"}
    P = "Explore the repo layout"

    # turn 1: main agent issues an Agent tool call with prompt P
    _rec(mgr, sid, [main_sys, {"role": "user", "content": "go"}], _agent_call(P), prompt_ids=[1, 2], output_ids=[10])
    # turn 2: sub-agent starts -- fresh system + user==P
    sub_sys = {"role": "system", "content": "SUB-SYS-explore"}
    _rec(
        mgr,
        sid,
        [sub_sys, {"role": "user", "content": P}],
        {"role": "assistant", "content": "s1"},
        prompt_ids=[20, 21],
        output_ids=[30],
    )

    samples = mgr.get_trajectory(sid, base_sample=_base())
    idns = _identities(samples)
    mains = [i for i in idns if i["origin"] == "main"]
    subs = [i for i in idns if i["origin"] == "sub_agent"]
    assert len(mains) == 1 and len(subs) == 1
    caller_node_id = mains[0]["node_list"][0]
    sub = subs[0]
    assert sub["match_kind"] == "exact"
    assert sub["caller_node_ids"] == [caller_node_id]
    assert sub["is_compact_start"] is False


# ---------------------------------------------------------------------------
# compact start
# ---------------------------------------------------------------------------


def test_compact_start_main():
    mgr = TrajectoryManager()
    sid = "compact"
    main_sys = {"role": "system", "content": "MAIN-SYS"}
    _rec(
        mgr,
        sid,
        [main_sys, {"role": "user", "content": "go"}],
        {"role": "assistant", "content": "a1"},
        prompt_ids=[1, 2],
        output_ids=[10],
    )
    # compaction: fresh history, same system, user begins with the summary prefix
    summary = COMPACT_SUMMARY_PREFIX + " ... summary body"
    _rec(
        mgr,
        sid,
        [main_sys, {"role": "user", "content": summary}],
        {"role": "assistant", "content": "a2"},
        prompt_ids=[1, 50, 51],
        output_ids=[60],
    )

    samples = mgr.get_trajectory(sid, base_sample=_base())
    idns = _identities(samples)
    compacts = [i for i in idns if i["is_compact_start"]]
    assert len(compacts) == 1
    assert compacts[0]["origin"] == "main"
    # the pre-compact turn is NOT a compact start
    assert any((not i["is_compact_start"]) and i["origin"] == "main" for i in idns)


# ---------------------------------------------------------------------------
# sub-agent that itself compacts (orthogonal) -- the turn25 shape
# ---------------------------------------------------------------------------


def test_sub_agent_internal_compact_orthogonal():
    mgr = TrajectoryManager()
    sid = "subcompact"
    main_sys = {"role": "system", "content": "MAIN-SYS"}
    P = "deep dive task"
    _rec(mgr, sid, [main_sys, {"role": "user", "content": "go"}], _agent_call(P), prompt_ids=[1, 2], output_ids=[10])
    sub_sys = {"role": "system", "content": "SUB-SYS-explore"}
    # sub-agent post-compact restart: sub system + user begins with summary prefix.
    summary = COMPACT_SUMMARY_PREFIX + " sub summary"
    _rec(
        mgr,
        sid,
        [sub_sys, {"role": "user", "content": summary}],
        {"role": "assistant", "content": "s1"},
        prompt_ids=[20, 70, 71],
        output_ids=[80],
    )

    samples = mgr.get_trajectory(sid, base_sample=_base())
    idns = _identities(samples)
    sub = next(i for i in idns if i["origin"] == "sub_agent")
    assert sub["origin"] == "sub_agent"
    assert sub["is_compact_start"] is True
    # compaction severs the token path -> caller link lost (empty), per spec
    assert sub["caller_node_ids"] == []


# ---------------------------------------------------------------------------
# parallel fan-out: same prompt issued by two caller nodes
# ---------------------------------------------------------------------------


def test_fanout_same_prompt_records_all_callers():
    mgr = TrajectoryManager()
    sid = "fanout"
    main_sys = {"role": "system", "content": "MAIN-SYS"}
    P = "same fanout prompt"
    # two main turns, each issuing the SAME agent prompt
    _rec(mgr, sid, [main_sys, {"role": "user", "content": "go"}], _agent_call(P), prompt_ids=[1, 2], output_ids=[10])
    _rec(
        mgr,
        sid,
        [main_sys, {"role": "user", "content": "go"}, _agent_call(P), {"role": "tool", "content": "r"}],
        _agent_call(P),
        prompt_ids=[1, 2, 10, 3],
        output_ids=[11],
    )
    # one sub-agent with prompt P -> should list BOTH caller node ids
    sub_sys = {"role": "system", "content": "SUB-SYS"}
    _rec(
        mgr,
        sid,
        [sub_sys, {"role": "user", "content": P}],
        {"role": "assistant", "content": "s"},
        prompt_ids=[20, 21],
        output_ids=[30],
    )

    samples = mgr.get_trajectory(sid, base_sample=_base())
    sub = next(i for i in _identities(samples) if i["origin"] == "sub_agent")
    assert sub["match_kind"] == "exact"
    assert len(sub["caller_node_ids"]) == 2


# ---------------------------------------------------------------------------
# approx fallback: trailing whitespace on the replayed prompt
# ---------------------------------------------------------------------------


def test_sub_agent_approx_match_whitespace():
    mgr = TrajectoryManager()
    sid = "approx"
    main_sys = {"role": "system", "content": "MAIN-SYS"}
    P = "task body"
    _rec(mgr, sid, [main_sys, {"role": "user", "content": "go"}], _agent_call(P), prompt_ids=[1, 2], output_ids=[10])
    sub_sys = {"role": "system", "content": "SUB-SYS"}
    _rec(
        mgr,
        sid,
        [sub_sys, {"role": "user", "content": P + "   "}],
        {"role": "assistant", "content": "s"},
        prompt_ids=[20, 21],
        output_ids=[30],
    )

    samples = mgr.get_trajectory(sid, base_sample=_base())
    sub = next(i for i in _identities(samples) if i["origin"] == "sub_agent")
    assert sub["match_kind"] == "approx"
    assert len(sub["caller_node_ids"]) == 1


# ---------------------------------------------------------------------------
# node_id uniqueness + cross-sid isolation
# ---------------------------------------------------------------------------


def test_node_ids_unique_and_per_sid():
    mgr = TrajectoryManager()
    for sid in ("A", "B"):
        sysm = {"role": "system", "content": "S"}
        _rec(
            mgr,
            sid,
            [sysm, {"role": "user", "content": "u"}],
            {"role": "assistant", "content": "a"},
            prompt_ids=[1, 2],
            output_ids=[10],
        )
        # node_ids assigned monotonically from 0 within each sid
        root = mgr._trees[sid]
        ids = [n.node_id for n in _iter(root) if n.node_id is not None]
        assert ids == sorted(ids)
        assert ids[0] == 0
        assert len(set(ids)) == len(ids)
    # draining A leaves B intact and B started its own node_id space at 0
    sa = mgr.get_trajectory("A", base_sample=_base())
    assert sa[0].metadata["identity"]["start_node_id"] == sa[0].metadata["identity"]["node_list"][0]


def _iter(root):
    stack = list(root.children)
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.children)
