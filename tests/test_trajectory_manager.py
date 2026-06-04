"""Unit tests for src_v2.trajectory_manager (Plan C: token-faithful).

What we test:
  (1) DFS merge only on (role, node_match_key): same prefix in messages
      space always lands on the same path regardless of prompt_ids drift.
  (2) get_trajectory linearization: turn 1 = full prompt + response;
      turn k>=2 = LCP-aligned drop-and-replace; tokens / loss_mask /
      logprobs all stay in sync.
  (3) TITO drift handling: when turn k+1.prompt diverges mid-stream from
      cumulative tokens, drift suffix is dropped, the new turn's prompt
      tail is appended, and the next response stays valid.
"""

from __future__ import annotations

import json
import logging

from slime.agent.trajectory_manager import (  # noqa: E402
    Node,
    TrajectoryManager,
    _group_messages_by_role,
    _lcp_len,
    node_match_key,
)
from slime.utils.types import Sample  # noqa: E402


# ---------------------------------------------------------------------------
# Helper-level tests
# ---------------------------------------------------------------------------


def test_node_match_key_is_dict_internal_sort_only():
    a = [{"role": "u", "content": "x"}]
    b = [{"content": "x", "role": "u"}]
    assert node_match_key(a) == node_match_key(b)

    c = [{"role": "u", "content": "x"}, {"role": "u", "content": "y"}]
    d = [{"role": "u", "content": "y"}, {"role": "u", "content": "x"}]
    assert node_match_key(c) != node_match_key(d)

    e = [{"role": "assistant", "tool_calls": [{"id": "1", "type": "function"}]}]
    f = [{"role": "assistant", "tool_calls": [{"type": "function", "id": "1"}]}]
    assert node_match_key(e) == node_match_key(f)
    print("PASS test_node_match_key_is_dict_internal_sort_only")


def test_group_messages_by_role_basic():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u1"}]
    groups = _group_messages_by_role(msgs)
    assert [g.role for g in groups] == ["system", "user"]
    assert [len(g.messages) for g in groups] == [1, 1]
    print("PASS test_group_messages_by_role_basic")


def test_group_messages_by_role_merges_adjacent_same_role():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
        {"role": "tool", "content": "t1"},
        {"role": "tool", "content": "t2"},
        {"role": "assistant", "content": "a2"},
    ]
    groups = _group_messages_by_role(msgs)
    assert [g.role for g in groups] == ["system", "user", "assistant", "tool", "assistant"]
    assert len(groups[3].messages) == 2
    print("PASS test_group_messages_by_role_merges_adjacent_same_role")


def test_lcp_len():
    assert _lcp_len([], []) == 0
    assert _lcp_len([1, 2, 3], []) == 0
    assert _lcp_len([], [1, 2, 3]) == 0
    assert _lcp_len([1, 2, 3], [1, 2, 3]) == 3
    assert _lcp_len([1, 2, 3], [1, 2, 4]) == 2
    assert _lcp_len([1, 2, 3, 4, 5], [1, 2, 3]) == 3
    print("PASS test_lcp_len")


# ---------------------------------------------------------------------------
# Fake tokenizer (kept only as a shape-matching prompt/response generator;
# trajectory_manager doesn't invoke it under plan C).
# ---------------------------------------------------------------------------


class FakeTokenizer:
    ROLE_START = {"system": 9001, "user": 9002, "assistant": 9003, "tool": 9004}
    ROLE_END = {"system": 9101, "user": 9102, "assistant": 9103, "tool": 9104}

    def apply_chat_template(self, messages, *, tools=None, add_generation_prompt=False, **kwargs):
        out: list[int] = []
        for m in messages:
            role = m["role"]
            content = m.get("content") or ""
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            out.append(self.ROLE_START[role])
            out.extend(ord(c) for c in content)
            out.append(self.ROLE_END[role])
        if add_generation_prompt:
            out.append(self.ROLE_START["assistant"])
        return out


def _render_prompt(messages, tools=None, tokenizer=None):
    tok = tokenizer or FakeTokenizer()
    return tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True)


def _render_response(content_str, tokenizer=None):
    tok = tokenizer or FakeTokenizer()
    return [ord(c) for c in content_str] + [tok.ROLE_END["assistant"]]


# ---------------------------------------------------------------------------
# Plan-C semantics tests
# ---------------------------------------------------------------------------


SYSTEM_MSG = "You are a python coding agent."
TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Run python code.",
            "parameters": {"type": "object", "properties": {"code": {"type": "string"}}},
        },
    },
]


def _three_turn_session(tok):
    """3-turn linear session via append_turn. Returns mgr, sid, per-turn (p,r)."""
    mgr = TrajectoryManager(tokenizer=tok)
    sid = "three-turn"

    sys_msg = {"role": "system", "content": SYSTEM_MSG}
    user1 = {"role": "user", "content": "Compute 2+2."}
    asst1 = {"role": "assistant", "content": "Computing."}
    tool1 = {"role": "tool", "content": "4"}
    asst2 = {"role": "assistant", "content": "Answer is 4."}

    p1 = _render_prompt([sys_msg, user1], tokenizer=tok)
    r1 = _render_response("Computing.", tokenizer=tok)
    mgr.append_turn(
        sid, prompt_messages=[sys_msg, user1], tools=TOOLS_OPENAI,
        prompt_ids=p1, response_ids=r1,
        response_logprobs=[-0.5] * len(r1),
        response_message=asst1, finish_reason="tool_calls",
    )

    p2 = _render_prompt([sys_msg, user1, asst1, tool1], tokenizer=tok)
    r2 = _render_response("Answer is 4.", tokenizer=tok)
    mgr.append_turn(
        sid, prompt_messages=[sys_msg, user1, asst1, tool1], tools=TOOLS_OPENAI,
        prompt_ids=p2, response_ids=r2,
        response_logprobs=[-0.4] * len(r2),
        response_message=asst2, finish_reason="stop",
    )
    return mgr, sid, [(p1, r1), (p2, r2)]


def test_append_single_turn_shapes_tree():
    tok = FakeTokenizer()
    mgr = TrajectoryManager(tokenizer=tok)
    sid = "single"
    sys_msg = {"role": "system", "content": "S"}
    user1 = {"role": "user", "content": "u"}
    p = _render_prompt([sys_msg, user1], tokenizer=tok)
    r = _render_response("a", tokenizer=tok)
    mgr.append_turn(
        sid, prompt_messages=[sys_msg, user1], tools=None,
        prompt_ids=p, response_ids=r, response_logprobs=None,
        response_message={"role": "assistant", "content": "a"},
        finish_reason="stop",
    )
    chain = list(mgr._trees[sid].leaves())[0].path_from_root()
    roles = [n.role for n in chain]
    assert roles == ["system", "user", "assistant"], roles
    asst = chain[-1]
    assert asst.turn_index == 1
    assert asst.turn_prompt_ids == p
    assert asst.turn_response_ids == r
    assert asst.turn_finish_reason == "stop"
    print("PASS test_append_single_turn_shapes_tree")


def test_append_three_turn_chain_no_fork():
    """3-turn session with consistent prompts -> exactly 1 leaf."""
    tok = FakeTokenizer()
    mgr, sid, _ = _three_turn_session(tok)
    leaves = [leaf for leaf in mgr._trees[sid].leaves() if not leaf.is_root]
    assert len(leaves) == 1
    chain = leaves[0].path_from_root()
    roles = [n.role for n in chain]
    assert roles == ["system", "user", "assistant", "tool", "assistant"], roles
    assert mgr.turn_count(sid) == 2
    print("PASS test_append_three_turn_chain_no_fork")


def test_fork_on_text_diff():
    """Different user content under shared sys -> 2 leaves, sys shared."""
    tok = FakeTokenizer()
    mgr = TrajectoryManager(tokenizer=tok)
    sid = "fork-text"
    sys_msg = {"role": "system", "content": "S"}

    for content in ["uA", "uB"]:
        user = {"role": "user", "content": content}
        p = _render_prompt([sys_msg, user], tokenizer=tok)
        mgr.append_turn(
            sid, prompt_messages=[sys_msg, user], tools=None,
            prompt_ids=p, response_ids=_render_response(content[-1], tokenizer=tok),
            response_logprobs=None,
            response_message={"role": "assistant", "content": content[-1]},
            finish_reason="stop",
        )

    root = mgr._trees[sid]
    assert len(root.children) == 1, "sys node must be shared"
    sys_node = root.children[0]
    assert len(sys_node.children) == 2, "user level must fork"
    leaves = [leaf for leaf in root.leaves() if not leaf.is_root]
    assert len(leaves) == 2
    print("PASS test_fork_on_text_diff")


def test_no_fork_on_token_only_diff():
    """Plan C: same text but tampered prompt_ids -> NO fork (DFS ignores tokens).

    This is the load-bearing behavior change vs the old prefix-match design.
    """
    tok = FakeTokenizer()
    mgr = TrajectoryManager(tokenizer=tok)
    sid = "tokens-diff-only"
    sys_msg = {"role": "system", "content": "S"}
    user1 = {"role": "user", "content": "u"}
    pa = _render_prompt([sys_msg, user1], tokenizer=tok)
    mgr.append_turn(
        sid, prompt_messages=[sys_msg, user1], tools=None,
        prompt_ids=pa, response_ids=_render_response("a", tokenizer=tok),
        response_logprobs=None,
        response_message={"role": "assistant", "content": "a"},
        finish_reason="stop",
    )
    tampered = list(pa)
    tampered[1] = tampered[1] ^ 1
    mgr.append_turn(
        sid, prompt_messages=[sys_msg, user1], tools=None,
        prompt_ids=tampered, response_ids=_render_response("b", tokenizer=tok),
        response_logprobs=None,
        response_message={"role": "assistant", "content": "b"},
        finish_reason="stop",
    )
    root = mgr._trees[sid]
    # Same (sys, user) path -> shared, but two different assistant turns
    # produce two assistant leaves under the same user node.
    assert len(root.children) == 1
    sys_node = root.children[0]
    assert len(sys_node.children) == 1
    user_node = sys_node.children[0]
    assert len(user_node.children) == 2, "two distinct assistant turns hang off shared user"
    leaves = [leaf for leaf in root.leaves() if not leaf.is_root]
    assert len(leaves) == 2
    print("PASS test_no_fork_on_token_only_diff")


def test_cross_sid_isolation():
    tok = FakeTokenizer()
    mgr = TrajectoryManager(tokenizer=tok)
    sys_msg = {"role": "system", "content": "S"}
    for sid, content in [("sid-a", "uA"), ("sid-b", "uB")]:
        user = {"role": "user", "content": content}
        p = _render_prompt([sys_msg, user], tokenizer=tok)
        mgr.append_turn(
            sid, prompt_messages=[sys_msg, user], tools=None,
            prompt_ids=p, response_ids=_render_response(content[-1], tokenizer=tok),
            response_logprobs=None,
            response_message={"role": "assistant", "content": content[-1]},
            finish_reason="stop",
        )
    assert len(list(mgr._trees["sid-a"].leaves())) == 1
    assert len(list(mgr._trees["sid-b"].leaves())) == 1
    print("PASS test_cross_sid_isolation")


def test_role_tool_in_chain():
    tok = FakeTokenizer()
    mgr = TrajectoryManager(tokenizer=tok)
    sid = "tool-chain"
    sys_msg = {"role": "system", "content": "S"}
    user1 = {"role": "user", "content": "u"}
    asst1 = {"role": "assistant", "content": "a1"}
    tool_a = {"role": "tool", "content": "tA"}
    tool_b = {"role": "tool", "content": "tB"}
    asst2 = {"role": "assistant", "content": "a2"}

    p1 = _render_prompt([sys_msg, user1], tokenizer=tok)
    r1 = _render_response("a1", tokenizer=tok)
    p2 = _render_prompt([sys_msg, user1, asst1, tool_a, tool_b], tokenizer=tok)
    r2 = _render_response("a2", tokenizer=tok)
    mgr.append_turn(sid, prompt_messages=[sys_msg, user1], tools=None,
                    prompt_ids=p1, response_ids=r1, response_logprobs=None,
                    response_message=asst1, finish_reason="stop")
    mgr.append_turn(sid, prompt_messages=[sys_msg, user1, asst1, tool_a, tool_b],
                    tools=None,
                    prompt_ids=p2, response_ids=r2, response_logprobs=None,
                    response_message=asst2, finish_reason="stop")

    chain = list(mgr._trees[sid].leaves())[0].path_from_root()
    roles = [n.role for n in chain]
    assert roles == ["system", "user", "assistant", "tool", "assistant"], roles
    assert len(chain[3].messages) == 2
    print("PASS test_role_tool_in_chain")


def test_response_logprobs_length_mismatch_raises():
    tok = FakeTokenizer()
    mgr = TrajectoryManager(tokenizer=tok)
    sys_msg = {"role": "system", "content": "S"}
    user1 = {"role": "user", "content": "u"}
    p = _render_prompt([sys_msg, user1], tokenizer=tok)
    try:
        mgr.append_turn("x", prompt_messages=[sys_msg, user1], tools=None,
                        prompt_ids=p, response_ids=[1, 2, 3],
                        response_logprobs=[-0.1, -0.2],
                        response_message={"role": "assistant", "content": ""},
                        finish_reason="stop")
    except ValueError as e:
        assert "response_logprobs length" in str(e)
        print("PASS test_response_logprobs_length_mismatch_raises")
        return
    raise AssertionError("expected ValueError")


def test_response_ids_empty_ok():
    tok = FakeTokenizer()
    mgr = TrajectoryManager(tokenizer=tok)
    sys_msg = {"role": "system", "content": "S"}
    user1 = {"role": "user", "content": "u"}
    p = _render_prompt([sys_msg, user1], tokenizer=tok)
    mgr.append_turn("x", prompt_messages=[sys_msg, user1], tools=None,
                    prompt_ids=p, response_ids=[], response_logprobs=None,
                    response_message=None, finish_reason="stop")
    chain = list(mgr._trees["x"].leaves())[0].path_from_root()
    asst = chain[-1]
    assert asst.role == "assistant"
    assert asst.turn_response_ids == []
    assert asst.turn_prompt_ids == p
    assert asst.messages == []
    print("PASS test_response_ids_empty_ok")


# ---------------------------------------------------------------------------
# get_trajectory linearization (Plan C heart of the matter)
# ---------------------------------------------------------------------------


def test_get_trajectory_single_turn():
    tok = FakeTokenizer()
    mgr = TrajectoryManager(tokenizer=tok)
    sid = "g1"
    sys_msg = {"role": "system", "content": "S"}
    user = {"role": "user", "content": "u"}
    p = _render_prompt([sys_msg, user], tokenizer=tok)
    r = _render_response("a", tokenizer=tok)
    mgr.append_turn(
        sid, prompt_messages=[sys_msg, user], tools=TOOLS_OPENAI,
        prompt_ids=p, response_ids=r, response_logprobs=[-0.5] * len(r),
        response_message={"role": "assistant", "content": "a"},
        finish_reason="stop",
    )
    samples = mgr.get_trajectory(sid, base_sample=Sample(index=7, prompt="hi"), reward=1.0)
    assert len(samples) == 1
    s = samples[0]
    assert s.tokens == p + r
    assert s.loss_mask == [0] * len(p) + [1] * len(r)
    assert s.rollout_log_probs == [0.0] * len(p) + [-0.5] * len(r)
    assert s.response_length == len(r)
    assert s.reward == 1.0
    assert s.metadata["finish_reason"] == "stop"
    assert s.metadata["tools"] == TOOLS_OPENAI
    assert "tito_dropped_tokens" not in s.metadata
    assert "tito_dropped_turns" not in s.metadata
    print("PASS test_get_trajectory_single_turn")


def test_get_trajectory_clean_multiturn():
    """Clean 2-turn session (no drift) linearizes as turn1 prompt+resp then
    turn2 (prompt - LCP) + resp, with full coherent loss_mask / logprobs."""
    tok = FakeTokenizer()
    mgr, sid, turns = _three_turn_session(tok)
    samples = mgr.get_trajectory(sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 1
    s = samples[0]

    (p1, r1), (p2, r2) = turns
    # LCP(p1+r1, p2) should equal len(p1)+len(r1) for our clean fake tokenizer
    # (p2 starts exactly with p1 contents + asst response + tool block + new gen prompt)
    L = _lcp_len(p1 + r1, p2)
    assert L == len(p1) + len(r1), f"clean session LCP should equal cumulative, got {L}"
    expected_tokens = p1 + r1 + p2[L:] + r2
    expected_loss = [0] * len(p1) + [1] * len(r1) + [0] * (len(p2) - L) + [1] * len(r2)
    expected_logp = [0.0] * len(p1) + [-0.5] * len(r1) + [0.0] * (len(p2) - L) + [-0.4] * len(r2)
    assert s.tokens == expected_tokens
    assert s.loss_mask == expected_loss
    assert s.rollout_log_probs == expected_logp
    assert s.response_length == len(r1) + len(r2)
    assert "tito_dropped_tokens" not in s.metadata
    assert "tito_dropped_turns" not in s.metadata
    print("PASS test_get_trajectory_clean_multiturn")


def test_get_trajectory_tito_drift_drops_and_replaces():
    """Plan C heart: turn 2 prompt diverges mid-stream from cumulative.
    Drop drift suffix (incl. turn 1 response tail), append turn 2 prompt[LCP:]
    as loss_mask=0, then turn 2 response."""
    tok = FakeTokenizer()
    mgr = TrajectoryManager(tokenizer=tok)
    sid = "tito"
    sys_msg = {"role": "system", "content": "S"}
    user = {"role": "user", "content": "u"}
    asst1 = {"role": "assistant", "content": "a1"}
    tool = {"role": "tool", "content": "t"}
    asst2 = {"role": "assistant", "content": "a2"}

    p1 = _render_prompt([sys_msg, user], tokenizer=tok)
    r1 = _render_response("a1", tokenizer=tok)
    mgr.append_turn(
        sid, prompt_messages=[sys_msg, user], tools=None,
        prompt_ids=p1, response_ids=r1, response_logprobs=[-0.5] * len(r1),
        response_message=asst1, finish_reason="tool_calls",
    )
    # Build turn 2 prompt the "honest" way (what chat template would emit),
    # then INJECT a synthetic divergence inside the assistant response region
    # — simulating Qwen3 chat template adding `</think>\n\n` to a re-rendered
    # assistant message. We splice 3 fake tokens past LCP so the manager
    # MUST drop part of cumulative and accept the template's version.
    p2_honest = _render_prompt([sys_msg, user, asst1, tool], tokenizer=tok)
    # Find a divergence position inside r1's region (cumulative offset
    # len(p1) + 1 happens to sit inside r1 which is "a1" + 9103 -> 3 tokens).
    drift_at = len(p1) + 1  # inside r1
    p2 = list(p2_honest)
    # Splice a 3-token "phantom" right where chat template would emit fake
    # `</think>\n\n` for the materialized assistant turn. Use sentinels.
    p2 = p2[:drift_at] + [77777, 77778, 77779] + p2[drift_at:]
    r2 = _render_response("a2", tokenizer=tok)
    mgr.append_turn(
        sid, prompt_messages=[sys_msg, user, asst1, tool], tools=None,
        prompt_ids=p2, response_ids=r2, response_logprobs=[-0.4] * len(r2),
        response_message=asst2, finish_reason="stop",
    )

    samples = mgr.get_trajectory(sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 1
    s = samples[0]

    # Manually compute the expected linearization:
    cumulative_after_t1 = p1 + r1
    L = _lcp_len(cumulative_after_t1, p2)
    # LCP should be drift_at (since first divergence is at drift_at)
    assert L == drift_at, f"LCP={L} expected drift_at={drift_at}"
    drift = len(cumulative_after_t1) - L  # how much of turn 1 we drop
    expected_tokens = cumulative_after_t1[:L] + p2[L:] + r2
    # loss_mask of cumulative_after_t1[:L]: first len(p1) are 0 (prompt),
    # the (L - len(p1)) into r1 region are 1 (response).
    in_r1_kept = L - len(p1)
    assert in_r1_kept >= 0
    expected_loss = (
        [0] * len(p1) + [1] * in_r1_kept + [0] * (len(p2) - L) + [1] * len(r2)
    )
    expected_logp = (
        [0.0] * len(p1)
        + [-0.5] * in_r1_kept
        + [0.0] * (len(p2) - L)
        + [-0.4] * len(r2)
    )

    assert s.tokens == expected_tokens, (
        f"len got={len(s.tokens)} want={len(expected_tokens)}"
    )
    assert s.loss_mask == expected_loss
    assert s.rollout_log_probs == expected_logp
    assert s.response_length == in_r1_kept + len(r2)
    assert s.metadata.get("tito_dropped_tokens") == drift
    assert s.metadata.get("tito_dropped_turns") == 1
    print("PASS test_get_trajectory_tito_drift_drops_and_replaces")


def test_get_trajectory_tito_drift_logs_warning():
    """Drift case must emit a logger.warning about dropped tokens."""
    tok = FakeTokenizer()
    mgr = TrajectoryManager(tokenizer=tok)
    sid = "tito-log"
    sys_msg = {"role": "system", "content": "S"}
    user = {"role": "user", "content": "u"}
    asst1 = {"role": "assistant", "content": "a1"}
    tool = {"role": "tool", "content": "t"}
    asst2 = {"role": "assistant", "content": "a2"}

    p1 = _render_prompt([sys_msg, user], tokenizer=tok)
    r1 = _render_response("a1", tokenizer=tok)
    mgr.append_turn(sid, prompt_messages=[sys_msg, user], tools=None,
                    prompt_ids=p1, response_ids=r1, response_logprobs=None,
                    response_message=asst1, finish_reason="tool_calls")
    p2 = _render_prompt([sys_msg, user, asst1, tool], tokenizer=tok)
    p2 = p2[:len(p1) + 1] + [42, 43] + p2[len(p1) + 1:]
    r2 = _render_response("a2", tokenizer=tok)
    mgr.append_turn(sid, prompt_messages=[sys_msg, user, asst1, tool], tools=None,
                    prompt_ids=p2, response_ids=r2, response_logprobs=None,
                    response_message=asst2, finish_reason="stop")

    records: list[str] = []

    class _Cap(logging.Handler):
        def emit(self, rec):
            records.append(rec.getMessage())

    logger_mod = logging.getLogger("slime.agent.trajectory_manager")
    h = _Cap()
    logger_mod.addHandler(h)
    try:
        mgr.get_trajectory(sid, base_sample=Sample(index=0, prompt=""))
    finally:
        logger_mod.removeHandler(h)
    assert any("TITO drift" in m for m in records), records
    print("PASS test_get_trajectory_tito_drift_logs_warning")


def test_get_trajectory_two_leaves_share_reward():
    """Forked tree (2 leaves) -> reward split evenly."""
    tok = FakeTokenizer()
    mgr = TrajectoryManager(tokenizer=tok)
    sid = "split"
    sys_msg = {"role": "system", "content": "S"}
    for content in ["uA", "uB"]:
        user = {"role": "user", "content": content}
        p = _render_prompt([sys_msg, user], tokenizer=tok)
        r = _render_response(content[-1], tokenizer=tok)
        mgr.append_turn(
            sid, prompt_messages=[sys_msg, user], tools=None,
            prompt_ids=p, response_ids=r, response_logprobs=None,
            response_message={"role": "assistant", "content": content[-1]},
            finish_reason="stop",
        )
    samples = mgr.get_trajectory(sid, base_sample=Sample(index=0, prompt=""), reward=2.0)
    assert len(samples) == 2
    assert all(s.reward == 1.0 for s in samples)
    print("PASS test_get_trajectory_two_leaves_share_reward")


def test_drop_clears_sid():
    tok = FakeTokenizer()
    mgr, sid, _ = _three_turn_session(tok)
    assert mgr.has_session(sid)
    mgr.get_trajectory(sid, base_sample=Sample(index=0, prompt=""))
    assert not mgr.has_session(sid)
    assert mgr.get_trajectory(sid, base_sample=Sample(index=0, prompt="")) == []
    print("PASS test_drop_clears_sid")


def test_get_trajectory_keep_when_drop_false():
    tok = FakeTokenizer()
    mgr, sid, _ = _three_turn_session(tok)
    samples = mgr.get_trajectory(sid, base_sample=Sample(index=0, prompt=""), drop=False)
    assert len(samples) == 1
    assert mgr.has_session(sid)
    print("PASS test_get_trajectory_keep_when_drop_false")


def test_debug_dump_shape():
    from examples.coding_agent_rl.trajectory_manager_debug import dump_tree_json, dump_tree_txt

    tok = FakeTokenizer()
    mgr, sid, _ = _three_turn_session(tok)
    txt = dump_tree_txt(mgr, sid)
    assert isinstance(txt, str) and txt
    for needle in ("session=", "[system]", "[user]", "[assistant]", "[tool]", "turns=2"):
        assert needle in txt, f"missing {needle!r}"
    # Plan C: assistant rows show turn= / prompt_ids= / response_ids=
    assert "turn=1" in txt
    assert "turn=2" in txt
    assert "prompt_ids=" in txt
    assert "response_ids=" in txt

    j = dump_tree_json(mgr, sid)
    assert j["found"] is True and j["sid"] == sid and j["turns"] == 2
    assert j["nodes_total"] == 5

    miss = dump_tree_txt(mgr, "no-such")
    assert miss == "<no session: no-such>"
    miss_j = dump_tree_json(mgr, "no-such")
    assert miss_j == {"sid": "no-such", "found": False}
    print("PASS test_debug_dump_shape")


def test_get_trajectory_tito_snapshot_disabled_by_default():
    """Not passing tito_snapshot_min_loss_tokens => same shape as before:
    even a large drift just drops, no snapshot Sample emitted."""
    tok = FakeTokenizer()
    mgr = TrajectoryManager(tokenizer=tok)  # no snapshot kwarg

    sid = "snap-disabled"
    sys_msg = {"role": "system", "content": "S"}
    user = {"role": "user", "content": "u"}
    asst1 = {"role": "assistant", "content": "a1" * 600}
    tool = {"role": "tool", "content": "t"}
    asst2 = {"role": "assistant", "content": "a2"}

    p1 = _render_prompt([sys_msg, user], tokenizer=tok)
    r1 = _render_response("a1" * 600, tokenizer=tok)
    assert len(r1) > 1000, f"need >1000 loss tokens for the test, got {len(r1)}"
    mgr.append_turn(
        sid, prompt_messages=[sys_msg, user], tools=None,
        prompt_ids=p1, response_ids=r1, response_logprobs=[-0.5] * len(r1),
        response_message=asst1, finish_reason="tool_calls",
    )
    p2_honest = _render_prompt([sys_msg, user, asst1, tool], tokenizer=tok)
    drift_at = len(p1) + 1
    p2 = p2_honest[:drift_at] + [77777, 77778, 77779] + p2_honest[drift_at:]
    r2 = _render_response("a2", tokenizer=tok)
    mgr.append_turn(
        sid, prompt_messages=[sys_msg, user, asst1, tool], tools=None,
        prompt_ids=p2, response_ids=r2, response_logprobs=[-0.4] * len(r2),
        response_message=asst2, finish_reason="stop",
    )

    samples = mgr.get_trajectory(sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 1, f"snapshot off must emit exactly 1 sample, got {len(samples)}"
    s = samples[0]
    assert "tito_snapshot" not in s.metadata
    assert "tito_snapshots_emitted" not in s.metadata
    assert s.metadata.get("tito_dropped_turns") == 1
    assert s.metadata.get("tito_dropped_tokens") > 0
    print("PASS test_get_trajectory_tito_snapshot_disabled_by_default")


def test_get_trajectory_tito_snapshot_emits_when_loss_tokens_above_threshold():
    """Drift with loss_mask=1 tokens >= threshold => emit one snapshot Sample
    plus the regular main-leaf Sample. Snapshot mask is COMPLEMENTARY
    (1 only at positions [L:] that were originally 1, 0 elsewhere).
    Main-leaf Sample is bit-for-bit identical to the snapshot-off case."""
    tok = FakeTokenizer()
    mgr = TrajectoryManager(tokenizer=tok, tito_snapshot_min_loss_tokens=100)

    sid = "snap-emit"
    sys_msg = {"role": "system", "content": "S"}
    user = {"role": "user", "content": "u"}
    asst1 = {"role": "assistant", "content": "a" * 500}
    tool = {"role": "tool", "content": "t"}
    asst2 = {"role": "assistant", "content": "a2"}

    p1 = _render_prompt([sys_msg, user], tokenizer=tok)
    r1 = _render_response("a" * 500, tokenizer=tok)
    assert len(r1) > 100
    mgr.append_turn(
        sid, prompt_messages=[sys_msg, user], tools=None,
        prompt_ids=p1, response_ids=r1, response_logprobs=[-0.5] * len(r1),
        response_message=asst1, finish_reason="tool_calls",
    )
    # Splice a fake divergence right at len(p1), so the entire r1 sits in drift.
    p2_honest = _render_prompt([sys_msg, user, asst1, tool], tokenizer=tok)
    drift_at = len(p1)
    p2 = p2_honest[:drift_at] + [77777] + p2_honest[drift_at:]
    r2 = _render_response("a2", tokenizer=tok)
    mgr.append_turn(
        sid, prompt_messages=[sys_msg, user, asst1, tool], tools=None,
        prompt_ids=p2, response_ids=r2, response_logprobs=[-0.4] * len(r2),
        response_message=asst2, finish_reason="stop",
    )

    samples = mgr.get_trajectory(
        sid, base_sample=Sample(index=42, group_id=42, prompt="P", label="L"),
        reward=1.0,
    )
    assert len(samples) == 2, f"expected 1 snapshot + 1 main, got {len(samples)}"
    snap, main = samples

    # snapshot
    assert snap.metadata.get("tito_snapshot") is True
    assert snap.metadata.get("tito_snapshot_at_turn") == 2
    assert snap.tokens == p1 + r1
    assert snap.loss_mask == [0] * len(p1) + [1] * len(r1)
    assert snap.metadata.get("tito_snapshot_loss_tokens") == len(r1)
    assert snap.response_length == len(r1)
    assert snap.rollout_log_probs == [0.0] * len(p1) + [-0.5] * len(r1)
    assert snap.group_id == 42
    assert snap.reward == 1.0  # only 1 leaf -> full share

    # main leaf must match the snapshot-OFF baseline
    mgr_off = TrajectoryManager(tokenizer=tok)
    sid_off = "snap-off-baseline"
    mgr_off.append_turn(
        sid_off, prompt_messages=[sys_msg, user], tools=None,
        prompt_ids=p1, response_ids=r1, response_logprobs=[-0.5] * len(r1),
        response_message=asst1, finish_reason="tool_calls",
    )
    mgr_off.append_turn(
        sid_off, prompt_messages=[sys_msg, user, asst1, tool], tools=None,
        prompt_ids=p2, response_ids=r2, response_logprobs=[-0.4] * len(r2),
        response_message=asst2, finish_reason="stop",
    )
    baseline = mgr_off.get_trajectory(
        sid_off, base_sample=Sample(index=42, group_id=42, prompt="P", label="L"),
        reward=1.0,
    )[0]
    assert main.tokens == baseline.tokens
    assert main.loss_mask == baseline.loss_mask
    assert main.rollout_log_probs == baseline.rollout_log_probs
    # main keeps NO tito_dropped_* for the snapshotted drift
    assert "tito_dropped_tokens" not in main.metadata
    assert "tito_dropped_turns" not in main.metadata
    assert main.metadata.get("tito_snapshots_emitted") == 1
    print("PASS test_get_trajectory_tito_snapshot_emits_when_loss_tokens_above_threshold")


def test_get_trajectory_tito_snapshot_skipped_when_below_threshold():
    """Drift with loss_mask=1 tokens < threshold => no snapshot; old behavior preserved."""
    tok = FakeTokenizer()
    mgr = TrajectoryManager(tokenizer=tok, tito_snapshot_min_loss_tokens=10_000)

    sid = "snap-skip"
    sys_msg = {"role": "system", "content": "S"}
    user = {"role": "user", "content": "u"}
    asst1 = {"role": "assistant", "content": "a1"}
    tool = {"role": "tool", "content": "t"}
    asst2 = {"role": "assistant", "content": "a2"}

    p1 = _render_prompt([sys_msg, user], tokenizer=tok)
    r1 = _render_response("a1", tokenizer=tok)
    mgr.append_turn(
        sid, prompt_messages=[sys_msg, user], tools=None,
        prompt_ids=p1, response_ids=r1, response_logprobs=[-0.5] * len(r1),
        response_message=asst1, finish_reason="tool_calls",
    )
    p2_honest = _render_prompt([sys_msg, user, asst1, tool], tokenizer=tok)
    drift_at = len(p1) + 1
    p2 = p2_honest[:drift_at] + [77777, 77778, 77779] + p2_honest[drift_at:]
    r2 = _render_response("a2", tokenizer=tok)
    mgr.append_turn(
        sid, prompt_messages=[sys_msg, user, asst1, tool], tools=None,
        prompt_ids=p2, response_ids=r2, response_logprobs=[-0.4] * len(r2),
        response_message=asst2, finish_reason="stop",
    )

    samples = mgr.get_trajectory(sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 1, f"below threshold must emit exactly 1 sample, got {len(samples)}"
    s = samples[0]
    assert "tito_snapshot" not in s.metadata
    assert "tito_snapshots_emitted" not in s.metadata
    assert s.metadata.get("tito_dropped_turns") == 1
    assert s.metadata.get("tito_dropped_tokens") > 0
    print("PASS test_get_trajectory_tito_snapshot_skipped_when_below_threshold")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    test_node_match_key_is_dict_internal_sort_only()
    test_group_messages_by_role_basic()
    test_group_messages_by_role_merges_adjacent_same_role()
    test_lcp_len()
    test_append_single_turn_shapes_tree()
    test_append_three_turn_chain_no_fork()
    test_fork_on_text_diff()
    test_no_fork_on_token_only_diff()
    test_cross_sid_isolation()
    test_role_tool_in_chain()
    test_response_logprobs_length_mismatch_raises()
    test_response_ids_empty_ok()
    test_get_trajectory_single_turn()
    test_get_trajectory_clean_multiturn()
    test_get_trajectory_tito_drift_drops_and_replaces()
    test_get_trajectory_tito_drift_logs_warning()
    test_get_trajectory_tito_snapshot_disabled_by_default()
    test_get_trajectory_tito_snapshot_emits_when_loss_tokens_above_threshold()
    test_get_trajectory_tito_snapshot_skipped_when_below_threshold()
    test_get_trajectory_two_leaves_share_reward()
    test_drop_clears_sid()
    test_get_trajectory_keep_when_drop_false()
    test_debug_dump_shape()
    print("\nALL PLAN-C TESTS PASSED.")


if __name__ == "__main__":
    main()
