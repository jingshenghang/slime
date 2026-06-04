"""Per-role chunk-merging trajectory tree manager (C-plan: token-faithful).

Design (Plan C, 2026-06-03):

* The tree is a router only. DFS merge keys on ``(role, node_match_key)``
  alone — no prompt_ids prefix check. Same conversation prefix in
  ``messages`` space always lands on the same path, regardless of any
  chat_template re-tokenization drift across turns.

* Each assistant leaf stores the THIS-TURN sglang snapshot:
  ``turn_prompt_ids`` / ``turn_response_ids`` / ``turn_response_logprobs``
  / ``turn_finish_reason`` / ``turn_index``. Non-assistant nodes carry no
  token attribution at all.

* ``get_trajectory`` linearizes each leaf turn-by-turn using LCP-aligned
  drop-and-replace: the cumulative tokens emitted so far are clamped to
  the longest common prefix with the next turn's prompt; any prior tokens
  past that LCP (the TITO drift suffix, including the previous turn's
  response if it lands in the drift region) are DROPPED along with their
  logprobs, then ``prompt[LCP:]`` is appended as loss_mask=0 (chat
  template's authoritative re-rendering wins), then the current turn's
  ``response`` is appended as loss_mask=1 with real logprobs.

* Trade-off: previous-turn response tokens that fall inside the drift
  region lose their training signal. In exchange, the final tokens
  sequence matches what the live model actually conditioned on for every
  later turn — logprobs stay coherent, no duplicated-content forks, no
  reliance on chat_template being position-invariant.

* On drift, ``Sample.metadata`` records:
    ``tito_dropped_tokens`` — total tokens dropped across all turns
    ``tito_dropped_turns``  — number of turns that triggered a drop
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Iterator

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


# ===========================================================================
# Node
# ===========================================================================


class Node:
    """One node in the trajectory tree.

    Routing fields (every node):
        role, messages, parent, children, metadata

    Per-turn snapshot fields (assistant leaves only — None on non-assistant
    and on internal assistant nodes that aren't a turn's own leaf):
        turn_prompt_ids:        list[int]    sglang prompt as fed to /generate
        turn_response_ids:      list[int]    sglang output ids
        turn_response_logprobs: list[float]
        turn_finish_reason:     str | None
        turn_index:             int          1-based, monotonic per session
    """

    __slots__ = (
        # routing
        "role",
        "messages",
        "metadata",
        "parent",
        "children",
        # per-turn snapshot (assistant leaves)
        "turn_prompt_ids",
        "turn_response_ids",
        "turn_response_logprobs",
        "turn_finish_reason",
        "turn_index",
    )

    def __init__(
        self,
        *,
        role: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        parent: "Node | None" = None,
    ) -> None:
        self.role = role
        self.messages = list(messages or [])
        self.metadata = dict(metadata or {})
        self.parent: Node | None = parent
        self.children: list[Node] = []
        # per-turn snapshot
        self.turn_prompt_ids: list[int] | None = None
        self.turn_response_ids: list[int] | None = None
        self.turn_response_logprobs: list[float] | None = None
        self.turn_finish_reason: str | None = None
        self.turn_index: int | None = None

    @property
    def is_root(self) -> bool:
        return self.parent is None

    def add_child(self, child: "Node") -> "Node":
        child.parent = self
        self.children.append(child)
        return child

    def path_from_root(self) -> list["Node"]:
        """Ordered list of nodes from the first non-root ancestor down to self."""
        chain: list[Node] = []
        cur: Node | None = self
        while cur is not None and not cur.is_root:
            chain.append(cur)
            cur = cur.parent
        chain.reverse()
        return chain

    def leaves(self) -> Iterator["Node"]:
        if not self.children:
            yield self
            return
        for c in self.children:
            yield from c.leaves()


# ===========================================================================
# node_match_key + role-grouping helpers
# ===========================================================================


def node_match_key(messages: list[dict[str, Any]]) -> str:
    """Identity key for a node's message list.

    json.dumps(sort_keys=True) sorts dict-internal keys recursively; list
    element order is preserved (which is what we want: message order and
    tool_calls order are both semantically significant).
    """
    return json.dumps(messages, sort_keys=True, ensure_ascii=False)


@dataclass
class _PromptGroup:
    role: str
    messages: list[dict[str, Any]] = field(default_factory=list)


def _group_messages_by_role(
    messages: list[dict[str, Any]],
) -> list[_PromptGroup]:
    groups: list[_PromptGroup] = []
    for m in messages:
        role = m.get("role")
        if not isinstance(role, str):
            logger.warning("skipping message without string role: %r", m)
            continue
        if groups and groups[-1].role == role:
            groups[-1].messages.append(m)
        else:
            groups.append(_PromptGroup(role=role, messages=[m]))
    return groups


def _lcp_len(a: list[int], b: list[int]) -> int:
    """Length of the longest common prefix between two int lists."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


# ===========================================================================
# TrajectoryManager
# ===========================================================================


class TrajectoryManager:
    """Per-sid trajectory tree manager.

    See module docstring for the C-plan invariants. Each ``append_turn``
    mounts >=0 prompt nodes (under the deepest matching ancestor) + exactly
    1 assistant leaf carrying that turn's sglang snapshot.
    """

    def __init__(
        self,
        *,
        tokenizer=None,
        chat_template_kwargs: dict[str, Any] | None = None,
        end_of_turn_token_id: int | None = None,
        tito_snapshot_min_loss_tokens: int | None = None,
    ) -> None:
        # tokenizer / chat_template_kwargs are no longer load-bearing under
        # plan C, but the constructor signature is kept for callsite
        # compatibility. _tokenizer is retained for forward-compat (callers
        # constructing TrajectoryManager(tokenizer=tok) shouldn't break).
        self._tokenizer = tokenizer
        self._ct_kwargs: dict[str, Any] = dict(chat_template_kwargs or {})
        self._end_of_turn_token_id = end_of_turn_token_id
        # Drift-snapshot threshold (loss_mask=1 token count inside drift suffix).
        # None or <= 0 disables; behavior then matches the pre-feature output.
        self._snap_threshold: int | None = (
            tito_snapshot_min_loss_tokens
            if (tito_snapshot_min_loss_tokens is not None and tito_snapshot_min_loss_tokens > 0)
            else None
        )
        self._trees: dict[str, Node] = {}
        self._turn_count: dict[str, int] = {}

    # -------------------- public ------------------------------------------

    def has_session(self, sid: str) -> bool:
        return sid in self._trees

    def turn_count(self, sid: str) -> int:
        return self._turn_count.get(sid, 0)

    def append_turn(
        self,
        sid: str,
        *,
        prompt_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        prompt_ids: list[int],
        response_ids: list[int],
        response_logprobs: list[float] | None,
        response_message: dict[str, Any] | None,
        finish_reason: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not prompt_messages:
            logger.warning("append_turn(sid=%s): empty prompt_messages; skipping", sid)
            return
        if response_logprobs is not None and len(response_logprobs) != len(response_ids):
            raise ValueError(
                f"response_logprobs length {len(response_logprobs)} != "
                f"response_ids length {len(response_ids)}"
            )

        root = self._trees.get(sid)
        if root is None:
            root = Node()
            self._trees[sid] = root

        groups = _group_messages_by_role(prompt_messages)

        # Step 1: DFS by (role, node_match_key) ONLY. No prompt_ids check.
        cur = root
        i = 0
        while i < len(groups):
            g_key = node_match_key(groups[i].messages)
            match: Node | None = None
            for child in cur.children:
                if child.role == groups[i].role and node_match_key(child.messages) == g_key:
                    match = child
                    break
            if match is None:
                break
            cur = match
            i += 1

        # Step 2: mount remaining prompt groups as plain routing nodes.
        # Token attribution happens at get_trajectory time, not here.
        for g in groups[i:]:
            md: dict[str, Any] = {}
            if g.role == "system" and tools is not None and not self._first_system_already_set(cur):
                md["tools"] = list(tools)
            cur = cur.add_child(Node(role=g.role, messages=list(g.messages), metadata=md))

        # Step 3: assistant leaf with this turn's sglang snapshot.
        asst_messages = [response_message] if response_message is not None else []
        asst = Node(
            role="assistant",
            messages=asst_messages,
            metadata=dict(metadata or {}),
        )
        asst.turn_prompt_ids = list(prompt_ids)
        asst.turn_response_ids = list(response_ids)
        asst.turn_response_logprobs = (
            list(response_logprobs) if response_logprobs is not None else None
        )
        asst.turn_finish_reason = finish_reason
        asst.turn_index = self._turn_count.get(sid, 0) + 1
        cur.add_child(asst)

        self._turn_count[sid] = asst.turn_index

    def get_trajectory(
        self,
        sid: str,
        *,
        base_sample=None,
        reward: float = 0.0,
        extra_metadata: dict[str, Any] | None = None,
        drop: bool = True,
    ) -> list:
        """Linearize each leaf into a slime ``Sample`` using LCP drop-and-replace.

        For each leaf, walk root→leaf collecting assistant nodes in order.
        Start with tokens=[]. For each assistant turn k (1-based):

          1. ``p = asst.turn_prompt_ids``, ``r = asst.turn_response_ids``.
          2. If k == 1: emit all of ``p`` as loss_mask=0 (plus 0.0 logprobs),
             then ``r`` as loss_mask=1 with real logprobs.
          3. If k >= 2: compute ``L = LCP(tokens, p)``. Truncate tokens /
             loss_mask / logprobs to length L (DROP everything past L —
             that includes the previous turn's response tokens that fall
             in the drift region; logging tells you how much was dropped).
             Then append ``p[L:]`` (loss_mask=0) and ``r`` (loss_mask=1).

        When drift fires on at least one turn, the returned Sample's
        ``metadata`` gains ``tito_dropped_tokens`` (total tokens dropped
        across the leaf) and ``tito_dropped_turns`` (how many turns
        triggered a drop). Both keys are absent when no drift occurs.

        See module docstring for the rationale.
        """
        if base_sample is None:
            base_sample = Sample(index=0, prompt="")

        root = self._trees.get(sid)
        if root is None:
            return []
        leaves = [leaf for leaf in root.leaves() if not leaf.is_root]
        samples: list[Sample] = []
        for leaf in leaves:
            chain = leaf.path_from_root()
            asst_chain = [n for n in chain if n.role == "assistant"]

            tokens: list[int] = []
            loss_mask: list[int] = []
            logprobs: list[float] = []
            total_dropped = 0
            dropped_turns = 0

            for k, asst in enumerate(asst_chain, start=1):
                p = list(asst.turn_prompt_ids or [])
                r = list(asst.turn_response_ids or [])
                lp = (
                    list(asst.turn_response_logprobs)
                    if asst.turn_response_logprobs is not None
                    else None
                )

                if k == 1:
                    emit_prompt = p
                else:
                    L = _lcp_len(tokens, p)
                    drift = len(tokens) - L
                    if drift > 0:
                        logger.warning(
                            "get_trajectory(sid=%s leaf turn=%d): TITO drift detected, "
                            "dropping %d prior tokens (incl. previous-turn response) to "
                            "realign with this turn's prompt",
                            sid, asst.turn_index, drift,
                        )
                        total_dropped += drift
                        dropped_turns += 1
                        tokens = tokens[:L]
                        loss_mask = loss_mask[:L]
                        logprobs = logprobs[:L]
                    emit_prompt = p[L:]

                tokens.extend(emit_prompt)
                loss_mask.extend([0] * len(emit_prompt))
                logprobs.extend([0.0] * len(emit_prompt))

                tokens.extend(r)
                loss_mask.extend([1] * len(r))
                if lp is not None:
                    logprobs.extend(lp)
                else:
                    logprobs.extend([0.0] * len(r))

            response_length = sum(1 for m in loss_mask if m == 1)
            last_asst = asst_chain[-1] if asst_chain else None
            first_sys = next((n for n in chain if n.role == "system"), None)
            tools_meta = first_sys.metadata.get("tools") if first_sys else None
            md: dict[str, Any] = {
                **(base_sample.metadata or {}),
                **(extra_metadata or {}),
                "finish_reason": last_asst.turn_finish_reason if last_asst else None,
                "tools": tools_meta,
            }
            if total_dropped > 0:
                md["tito_dropped_tokens"] = total_dropped
                md["tito_dropped_turns"] = dropped_turns
            samples.append(
                Sample(
                    index=base_sample.index,
                    group_id=(
                        base_sample.group_id
                        if base_sample.group_id is not None
                        else base_sample.index
                    ),
                    prompt=base_sample.prompt,
                    label=base_sample.label,
                    tokens=tokens,
                    response_length=response_length,
                    loss_mask=loss_mask,
                    rollout_log_probs=logprobs,
                    reward=(reward / len(leaves)) if leaves else 0.0,
                    status=Sample.Status.COMPLETED,
                    metadata=md,
                )
            )
        if drop:
            self._trees.pop(sid, None)
            self._turn_count.pop(sid, None)
        return samples

    # -------------------- internals ----------------------------------------

    @staticmethod
    def _first_system_already_set(start: Node) -> bool:
        """Walk start->root looking for a system node already carrying tools."""
        cur: Node | None = start
        while cur is not None and not cur.is_root:
            if cur.role == "system" and cur.metadata.get("tools") is not None:
                return True
            cur = cur.parent
        return False


__all__ = [
    "Node",
    "TrajectoryManager",
    "node_match_key",
    "_group_messages_by_role",
    "_lcp_len",
]
