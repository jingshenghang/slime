"""Smoke tests for src_v2.trajectory_manager_debug.

Verifies the debug dumper can render an empty session, a single-turn session,
and a 2-turn session. Drives TrajectoryManager directly via append_turn so
we don't depend on middleware wiring.
"""

from __future__ import annotations


from examples.coding_agent_rl.trajectory_manager_debug import dump_tree_json, dump_tree_txt  # noqa: E402
from slime.agent.trajectory_manager import TrajectoryManager  # noqa: E402


def test_dump_missing_sid():
    mgr = TrajectoryManager()
    assert dump_tree_txt(mgr, "no-such") == "<no session: no-such>"
    assert dump_tree_json(mgr, "no-such") == {"sid": "no-such", "found": False}
    print("PASS test_dump_missing_sid")


def test_dump_single_turn():
    mgr = TrajectoryManager()
    sid = "s1"
    mgr.append_turn(
        sid,
        prompt_messages=[
            {"role": "system", "content": "S"},
            {"role": "user", "content": "u"},
        ],
        tools=None,
        prompt_ids=[1, 2, 3],
        response_ids=[10, 11],
        response_logprobs=[-0.5, -0.6],
        response_message={"role": "assistant", "content": "a"},
        finish_reason="stop",
    )
    txt = dump_tree_txt(mgr, sid)
    for needle in (
        "session=s1",
        "turns=1",
        "leaves=1",
        "[system]",
        "[user]",
        "[assistant]",
        "turn=1",
        "prompt_ids=(3)",
        "response_ids=(2)",
        "finish=stop",
        "has_logprobs=True",
    ):
        assert needle in txt, f"missing {needle!r} in:\n{txt}"

    j = dump_tree_json(mgr, sid)
    assert j["found"] is True
    assert j["sid"] == "s1"
    assert j["turns"] == 1
    assert j["leaves"] == 1
    assert j["nodes_total"] == 3
    assert isinstance(j["root"], dict)
    print("PASS test_dump_single_turn")


def test_dump_two_turns_chain():
    mgr = TrajectoryManager()
    sid = "s2"
    sys_m = {"role": "system", "content": "S"}
    user1 = {"role": "user", "content": "u"}
    asst1 = {"role": "assistant", "content": "a1"}
    tool1 = {"role": "tool", "content": "t"}
    asst2 = {"role": "assistant", "content": "a2"}
    p1 = [1, 2]
    r1 = [10, 11]
    mgr.append_turn(
        sid,
        prompt_messages=[sys_m, user1],
        tools=None,
        prompt_ids=p1,
        response_ids=r1,
        response_logprobs=None,
        response_message=asst1,
        finish_reason="tool_calls",
    )
    p2 = p1 + r1 + [20, 21]
    r2 = [30, 31]
    mgr.append_turn(
        sid,
        prompt_messages=[sys_m, user1, asst1, tool1],
        tools=None,
        prompt_ids=p2,
        response_ids=r2,
        response_logprobs=None,
        response_message=asst2,
        finish_reason="stop",
    )

    txt = dump_tree_txt(mgr, sid)
    assert "turns=2" in txt
    assert "[tool]" in txt
    assert "turn=2" in txt
    print("PASS test_dump_two_turns_chain")


def test_dump_reasoning_and_thinking():
    """reasoning_content (OpenAI shape) and Anthropic thinking blocks should
    both surface in the text summary; include_messages=True should embed raw
    messages so reasoning_content survives JSON round-trip."""
    mgr = TrajectoryManager()
    sid = "s3"
    mgr.append_turn(
        sid,
        prompt_messages=[
            {"role": "system", "content": "S"},
            # Anthropic-shape thinking block from a replayed prior assistant turn
            {
                "role": "user",
                "content": [
                    {"type": "thinking", "thinking": "user-side-think"},
                    {"type": "text", "text": "u"},
                ],
            },
        ],
        tools=None,
        prompt_ids=[1, 2, 3],
        response_ids=[10, 11],
        response_logprobs=None,
        # OpenAI shape: reasoning_content is a sibling of content
        response_message={
            "role": "assistant",
            "content": "a",
            "reasoning_content": "asst-side-think",
        },
        finish_reason="stop",
    )

    txt = dump_tree_txt(mgr, sid)
    assert "user-side-think" in txt, f"thinking block missing in:\n{txt}"
    assert "asst-side-think" in txt, f"reasoning_content missing in:\n{txt}"
    assert "reason:" in txt

    j = dump_tree_json(mgr, sid, include_messages=True)

    def collect_text(node, acc):
        acc.append(node.get("text", ""))
        for c in node.get("children", []):
            collect_text(c, acc)

    all_text: list[str] = []
    collect_text(j["root"], all_text)
    blob = "\n".join(all_text)
    assert "user-side-think" in blob
    assert "asst-side-think" in blob

    def find_assistant(node):
        if node.get("role") == "assistant":
            return node
        for c in node.get("children", []):
            r = find_assistant(c)
            if r is not None:
                return r
        return None

    asst = find_assistant(j["root"])
    assert asst is not None and "messages" in asst
    assert asst["messages"][0].get("reasoning_content") == "asst-side-think"

    # include_messages defaults off — no messages embedded
    j2 = dump_tree_json(mgr, sid)
    asst2 = find_assistant(j2["root"])
    assert asst2 is not None and "messages" not in asst2
    print("PASS test_dump_reasoning_and_thinking")


def main() -> None:
    test_dump_missing_sid()
    test_dump_single_turn()
    test_dump_two_turns_chain()
    test_dump_reasoning_and_thinking()
    print("\nALL trajectory_manager_debug tests PASSED")


if __name__ == "__main__":
    main()
