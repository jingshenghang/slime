"""Real-trajectory replay check for identity metadata.

Replays the wire ``/v1/messages`` requests of a recorded SWE rollout
(``0610-e2e-test/runs/20260614_134732/0016``) through TrajectoryManager and
asserts the identity facets resolve as observed by hand-inspection:

* the run mixes ``main`` and ``sub_agent`` origins;
* compaction is detected (``is_compact_start``) and is orthogonal to origin --
  some sub-agent turns are also post-compact restarts (the turn25 shape);
* at least one sub-agent turn links back to its caller node by exact prompt
  match, and post-compact sub-agent turns keep ``origin=sub_agent`` but lose the
  caller (compaction severs the token path).

The recorded turns are REQUESTS only (no responses), so each turn's response is
reconstructed from the next request's replayed history -- imperfect (it triggers
benign rewrite-forks) but enough to exercise all three facets end-to-end. The
test skips if the run directory is absent so it stays green off the dev box.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slime.agent.adapters.anthropic import _fold_mid_list_system_into_user, _translate_messages
from slime.agent.adapters.common import tool_call_dict  # noqa: F401  (kept: documents wire shape)
from slime.agent.trajectory import TrajectoryManager, TurnRecord
from slime.utils.types import Sample

RUN = Path("/mnt/jingshenghang/code/slime_swe/0610-e2e-test/runs/20260614_134732/0016")


def _load_requests() -> list[list[dict]]:
    """Translated chat-message lists for each real /v1/messages request."""
    out: list[list[dict]] = []
    for line in (RUN / "turns.jsonl").read_text().splitlines():
        t = json.loads(line)
        p = t.get("payload")
        if not isinstance(p, dict) or not (p.get("messages") or []):
            continue
        body = dict(p)
        _fold_mid_list_system_into_user(body)
        tr = _translate_messages(body.get("messages") or [], body.get("system"))
        if tr:
            out.append(tr)
    return out


def _reply_revealed_by_next(prev: list[dict], cur: list[dict]) -> dict:
    """The assistant reply to ``prev``'s turn, recovered from the next request."""
    n = 0
    while n < len(prev) and n < len(cur) and prev[n] == cur[n]:
        n += 1
    for m in cur[n:]:
        if m.get("role") == "assistant":
            return m
    for m in reversed(cur):
        if m.get("role") == "assistant":
            return m
    return {"role": "assistant", "content": ""}


@pytest.mark.skipif(not RUN.exists(), reason="recorded run dir not present")
def test_replay_0016_identity_facets():
    reqs = _load_requests()
    assert len(reqs) >= 10

    mgr = TrajectoryManager()
    sid = "cagent-0016"
    for i, tr in enumerate(reqs):
        rmsg = _reply_revealed_by_next(tr, reqs[i + 1]) if i + 1 < len(reqs) else {"role": "assistant", "content": ""}
        prompt_ids = [1_000_000 + i * 1000 + j for j in range(len(tr) + 1)]
        mgr.record_turn(
            sid,
            turn=TurnRecord(prompt_ids=prompt_ids, output_ids=[2_000_000 + i], finish_reason="stop"),
            prompt_messages=tr,
            response_message=rmsg,
        )

    samples = mgr.get_trajectory(sid, base_sample=Sample(index=0, group_index=0, prompt="p", label="l"))
    idns = [s.metadata["identity"] for s in samples]

    origins = {i["origin"] for i in idns}
    assert origins == {"main", "sub_agent"}, origins

    # compaction detected and orthogonal: some sub-agent turns are compact starts
    assert any(i["is_compact_start"] for i in idns)
    assert any(i["origin"] == "sub_agent" and i["is_compact_start"] for i in idns)
    assert any(i["origin"] == "main" and not i["is_compact_start"] for i in idns)

    # caller linkage: at least one sub-agent turn matched its caller exactly
    linked = [i for i in idns if i["origin"] == "sub_agent" and i["caller_node_ids"]]
    assert linked, "no sub_agent sample linked to a caller node"
    assert all(i["match_kind"] in ("exact", "approx") for i in linked)

    # post-compact sub-agent turns keep origin but drop the (severed) caller link
    severed = [i for i in idns if i["origin"] == "sub_agent" and i["is_compact_start"]]
    assert all(i["caller_node_ids"] == [] for i in severed)
