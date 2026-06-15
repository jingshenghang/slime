"""Build a per-session training trajectory from multi-turn conversation data.

The :class:`TrajectoryManager` builds one trajectory per session. ``record_turn``
feeds in each turn (prompt messages + the served model's sglang snapshot),
routing it into a per-sid message tree; ``get_trajectory`` then linearizes that
tree into a ``list[Sample]`` of loss-masked training rows, tolerating TITO
re-tokenization drift via fork/replace.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
from collections.abc import Iterator
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# Prefix Claude Code injects as the first user message of a post-compaction
# request (context ran out -> history replaced by a summary). Used as the
# content signal for ``is_compact_start`` (see ``_classify_segment``).
COMPACT_SUMMARY_PREFIX = "This session is being continued from a previous conversation"

# Tool-call names that spawn a sub-agent with a fresh context. The wire records
# the sub-agent's task text under ``input.prompt``; the manager sees it again as
# the first user message of the sub-agent's own segment, which is how a sub-agent
# is linked back to its caller (see ``_build_agent_prompt_index``).
SUBAGENT_TOOL_NAMES = ("Agent", "Task")


# ===========================================================================
# TurnRecord
# ===========================================================================


@dataclasses.dataclass(frozen=True)
class TurnRecord:
    """One sglang ``/generate`` snapshot: the contract between an adapter and the
    manager. Adapters build it from a turn's prompt/output token ids; ``record_turn``
    consumes it."""

    prompt_ids: list[int]
    output_ids: list[int]
    finish_reason: str
    output_log_probs: list[float] = dataclasses.field(default_factory=list)


# ===========================================================================
# MessageNode
# ===========================================================================


class MessageNode:
    """One node in a session's routing tree, carrying a single chat message
    (``None`` for the dummy root and for an assistant leaf we generated but
    whose ``response_message`` was empty).

    The two kinds are distinguished by whether ``turn`` is set, which reflects
    WHERE the message came from:

    * **generated** (``turn is not None``): an assistant message the model
      actually generated this turn, fed in via ``record_turn``. ``turn`` holds
      its :class:`TurnRecord` -- the prompt/output ids, logprobs and finish
      reason that ``get_trajectory`` linearizes into training tokens.
    * **routing-only** (``turn is None``): the message came from the prompt, not
      from generation, so it only exists to route. This is every
      system/user/tool node, AND any assistant we did NOT generate: a foreign
      assistant the client replayed in a later prompt, or a prior generated turn
      demoted by the rewrite-merge in ``_try_merge_assistant_rewrite``.
    """

    def __init__(
        self,
        *,
        role: str | None = None,
        message: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        parent: MessageNode | None = None,
    ) -> None:
        self.role = role
        self.message = message
        self.metadata = dict(metadata or {})
        self.parent: MessageNode | None = parent
        self.children: list[MessageNode] = []
        self.turn: TurnRecord | None = None  # the generated TurnRecord, else None (routing-only)
        self.turn_index: int | None = None
        # Per-sid monotonic id assigned at mount time to EVERY node (routing-only
        # and generated alike). Stable within one sid; reused as the unit a
        # Sample's ``identity.node_list`` references and a sub-agent's
        # ``caller_node_ids`` points at. The dummy root keeps ``None``.
        self.node_id: int | None = None
        # Shared by sibling leaf paths; the first to reach it trains on it, the rest
        # re-emit it as loss_mask=0 context -- so each response is trained exactly once.
        self.response_trained: bool = False

    @property
    def is_root(self) -> bool:
        return self.parent is None

    def add_child(self, child: MessageNode) -> MessageNode:
        child.parent = self
        self.children.append(child)
        return child

    def path_from_root(self) -> list[MessageNode]:
        """Ordered list of nodes from the first non-root ancestor down to self."""
        chain: list[MessageNode] = []
        node: MessageNode | None = self
        while node is not None and not node.is_root:
            chain.append(node)
            node = node.parent
        chain.reverse()
        return chain

    def leaves(self) -> Iterator[MessageNode]:
        if not self.children:
            yield self
            return
        for c in self.children:
            yield from c.leaves()


# ===========================================================================
# drift classification — how an incoming turn's prompt relates to held tokens
# ===========================================================================


def _common_prefix_len(a: list[int], b: list[int], chunk: int = 4096) -> int:
    limit = min(len(a), len(b))
    matched = 0
    while matched < limit:
        chunk_end = min(matched + chunk, limit)
        if a[matched:chunk_end] == b[matched:chunk_end]:
            matched = chunk_end
        else:
            while matched < chunk_end and a[matched] == b[matched]:
                matched += 1
            return matched
    return matched


class DriftKind(enum.Enum):
    CLEAN = "clean"  # drift == 0: prompt_ids exactly extends held tokens; append the tail beyond them
    REALIGN = "realign"  # drift inside the most-recent response span and short incoming response; replace that span (loss_mask=0)
    FORK = "fork"  # everything else: close this builder, open a fresh one as a fork


# ===========================================================================
# SampleBuilder — accumulates turns into one trainable Sample (fork closes it)
# ===========================================================================


class _SampleBuilder:
    """Accumulates a chain's turns into the token sequence of one ``Sample``.

    A chain of turns is appended one at a time via :meth:`append_turn`. Ideally
    each turn's prompt exactly extends the tokens we already hold, but a replayed
    turn rarely re-tokenizes byte-for-byte: TITO round-trips and chat-template
    re-rendering both perturb the ids of content we've already seen. The builder
    handles this drift in a source-agnostic way, classified by where and how far
    the prompt diverges from the held tokens (see :meth:`classify_token_drift`):

    * **CLEAN** -- no drift; append the prompt tail beyond what we hold.
    * **REALIGN** -- a short divergence inside the most-recent response span;
      overwrite that span from the prompt as loss_mask=0 and keep accumulating.
    * **FORK** -- divergence too large or too early to absorb; this builder is
      rejected and the caller closes it and opens a fresh one. That boundary is
      the "fork".

    Each surviving builder yields one Sample.
    """

    def __init__(self, fork_threshold: int) -> None:
        self._fork_threshold = fork_threshold
        self.tokens: list[int] = []
        self.loss_mask: list[int] = []
        self.logprobs: list[float] = []
        self.last_response_start_idx: int | None = None
        self.leading_prompt_len: int = 0
        # Generated assistant nodes packed into this builder, in append order.
        # "Include = record": a node re-emitted as loss=0 context (claimed by an
        # earlier sibling leaf) is still listed here, so ``node_list`` reflects
        # what this Sample spans; reverse lookup picks the trainer (loss=1).
        self.nodes: list[MessageNode] = []

    def classify_token_drift(self, turn: TurnRecord) -> DriftKind:
        """Decide how this builder should absorb ``turn``'s prompt.

        The incoming turn's prompt is expected to match the tokens this builder
        already holds as an exact prefix. When token drift has occurred -- the
        prompt diverges from the held tokens -- we decide whether to REALIGN
        (heal a short divergence inside the most-recent response span) or to FORK
        (``len(turn.output_ids) >= fork_threshold``, or the divergence sits too
        early to absorb). With no drift the turn is handled the CLEAN way -- a
        plain prefix extension.
        """
        realign_at = _common_prefix_len(self.tokens, turn.prompt_ids)
        drift = len(self.tokens) - realign_at

        if drift == 0:
            return DriftKind.CLEAN

        # REALIGN only heals drift that falls inside the most-recent response span
        # (and is short); divergence anywhere earlier, or an empty builder, forks.
        start = self.last_response_start_idx
        if start is not None and realign_at >= start and len(turn.output_ids) < self._fork_threshold:
            return DriftKind.REALIGN
        return DriftKind.FORK

    def append_turn(self, turn: TurnRecord, kind: DriftKind, *, trained: bool = True) -> None:
        """Append one turn into this SampleBuilder, branching on ``kind``: for REALIGN
        we overwrite the already-saved response span, for CLEAN we just append this
        turn's prompt tail."""
        assert kind is not DriftKind.FORK, "append_turn called on a builder that would fork"

        is_first_turn = self.last_response_start_idx is None

        # --- append this turn's prompt tail (loss_mask=0) ---
        if kind is DriftKind.REALIGN:
            self._align_to_prompt(turn.prompt_ids)  # drop the drifted tail, re-append from prompt
        else:  # CLEAN: held tokens are an exact prefix of prompt_ids; append the tail beyond them
            self._append_tokens(turn.prompt_ids[len(self.tokens) :], loss_mask=0)

        # --- append this turn's generated response (loss_mask=1 unless re-emitted as context) ---
        self.last_response_start_idx = len(self.tokens)
        self._append_tokens(
            turn.output_ids, loss_mask=int(trained), logprobs=turn.output_log_probs if trained else None
        )

        if is_first_turn:
            self.leading_prompt_len = len(turn.prompt_ids)

    def add_node(self, node: MessageNode) -> None:
        """Record the generated node whose turn was just appended (for node_list)."""
        self.nodes.append(node)

    def _align_to_prompt(self, prompt_ids: list[int]) -> None:
        """Heal REALIGN drift by overwriting the most-recent response span with
        ``prompt_ids`` as loss_mask=0: the drifted tokens carry no signal, and re-appending
        from the prompt keeps the builder contiguous. Earlier turns are untouched."""
        response_start = self.last_response_start_idx
        tail = prompt_ids[response_start:]
        self.tokens[response_start:] = tail
        self.loss_mask[response_start:] = [0] * len(tail)
        self.logprobs[response_start:] = [0.0] * len(tail)

    def _append_tokens(self, ids: list[int], *, loss_mask: int, logprobs: list[float] | None = None) -> None:
        self.tokens.extend(ids)
        self.loss_mask.extend([loss_mask] * len(ids))
        self.logprobs.extend(logprobs if logprobs else [0.0] * len(ids))

    def has_trained_response(self) -> bool:
        return any(self.loss_mask[self.leading_prompt_len :])

    def _identity_metadata(self) -> dict[str, Any]:
        """Build this Sample's ``identity`` block from the generated nodes it spans.

        The origin / compact / caller facts were resolved once per segment in
        ``_classify_segments`` and stamped onto each generated node's
        ``identity_segment``; here we just read the start node's segment and list
        the node ids this builder packed. A builder always begins at a segment
        start (a fork opens a fresh builder), so ``nodes[0]`` carries the
        authoritative identity for the whole Sample.
        """
        start = self.nodes[0]
        seg = start.metadata.get("identity_segment", {})
        return {
            "origin": seg.get("origin", "main"),
            "is_compact_start": bool(seg.get("is_compact_start", False)),
            "node_list": [n.node_id for n in self.nodes],
            "start_node_id": start.node_id,
            "caller_node_ids": list(seg.get("caller_node_ids", [])),
            "match_kind": seg.get("match_kind"),
        }

    def to_sample(self, base_sample: Sample, extra_metadata: dict[str, Any] | None) -> Sample:
        """Emit the accumulated tokens as one ``Sample``, stripping the first-turn
        prompt so loss_mask / logprobs cover only the response region."""
        start = self.leading_prompt_len  # first-turn prompt stripped; response region starts here
        metadata = dict(extra_metadata or {})
        if self.nodes:
            metadata["identity"] = self._identity_metadata()
        return Sample(
            index=base_sample.index,
            group_index=base_sample.group_index,
            rollout_id=base_sample.rollout_id if base_sample.rollout_id is not None else base_sample.index,
            prompt=base_sample.prompt,
            label=base_sample.label,
            tokens=list(self.tokens),
            response_length=len(self.loss_mask) - start,
            loss_mask=self.loss_mask[start:],
            rollout_log_probs=self.logprobs[start:],
            reward=0.0,
            status=Sample.Status.COMPLETED,
            metadata=metadata,
        )


# ===========================================================================
# TrajectoryManager
# ===========================================================================


class TrajectoryManager:
    def __init__(self, *, fork_threshold_tokens: int | None = None) -> None:
        self._fork_threshold: int = 1024 if fork_threshold_tokens is None else fork_threshold_tokens
        self._trees: dict[str, MessageNode] = {}
        self._turn_count: dict[str, int] = {}
        self._node_count: dict[str, int] = {}  # per-sid monotonic node_id allocator

    # -------------------- public ------------------------------------------

    def has_session(self, sid: str) -> bool:
        return sid in self._trees

    def turn_count(self, sid: str) -> int:
        return self._turn_count.get(sid, 0)

    def record_turn(
        self,
        sid: str,
        *,
        turn: TurnRecord,
        prompt_messages: list[dict[str, Any]],
        response_message: dict[str, Any] | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not prompt_messages:
            logger.warning("record_turn(sid=%s): empty prompt_messages; skipping", sid)
            return
        assert not turn.output_log_probs or len(turn.output_log_probs) == len(turn.output_ids), (
            f"turn.output_log_probs length {len(turn.output_log_probs)} != "
            f"turn.output_ids length {len(turn.output_ids)}"
        )

        root = self._trees.setdefault(sid, MessageNode())

        node, depth = self._find_mount_point(root, prompt_messages)
        node, depth = self._try_merge_assistant_rewrite(sid, node, prompt_messages, depth)
        node = self._mount_prompt_messages(node, prompt_messages[depth:], sid=sid)
        self._attach_assistant_leaf(sid, node, turn=turn, response_message=response_message, metadata=metadata)

    def get_trajectory(
        self,
        sid: str,
        *,
        base_sample: Sample,
        reward: float = 0.0,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[Sample]:
        """Linearize this sid's routing tree into slime ``Sample`` objects and
        consume the session.

        Each routing leaf yields one or more Samples; ``reward`` is split evenly
        across all of them. The sid is dropped afterwards, so a second call for
        the same sid returns ``[]``.
        """
        root = self._trees.get(sid)
        if root is None:
            return []

        # Resolve per-segment identity (main / sub_agent / compact-start + caller)
        # onto the generated nodes before draining, so each emitted Sample can read
        # its start node's identity. Pure annotation -- never affects routing.
        self._classify_segments(root)

        samples: list[Sample] = []
        for routing_leaf in root.leaves():
            if routing_leaf.is_root:
                continue
            chain = routing_leaf.path_from_root()
            samples.extend(self._chain_to_samples(chain, base_sample=base_sample, extra_metadata=extra_metadata))

        # TODO custom reward func
        per_sample_reward = (reward / len(samples)) if samples else 0.0
        for s in samples:
            s.reward = per_sample_reward

        self._trees.pop(sid, None)
        self._turn_count.pop(sid, None)
        self._node_count.pop(sid, None)
        return samples

    # -------------------- internals ----------------------------------------

    def _find_mount_point(self, root: MessageNode, messages: list[dict[str, Any]]) -> tuple[MessageNode, int]:
        """Walk down the tree matching each message by role and dict equality (==),
        returning the deepest node that still matches and where to mount the rest."""
        node = root
        depth = 0
        while depth < len(messages):
            msg = messages[depth]
            next_child = None
            for child in node.children:
                if child.role == msg.get("role") and child.message == msg:
                    next_child = child
                    break
            if next_child is None:
                break
            node = next_child
            depth += 1
        return node, depth

    def _try_merge_assistant_rewrite(
        self,
        sid: str,
        node: MessageNode,
        prompt_messages: list[dict[str, Any]],
        depth: int,
    ) -> tuple[MessageNode, int]:
        """Merge a short assistant-rewrite onto its node instead of forking.

        A harness may replay a prior assistant message slightly re-rendered (e.g.
        whitespace) in a later prompt. It no longer matches the node we generated,
        so it would fork -- stranding the original generated turn as a dead-end
        leaf that still emits its own training Sample. Instead we overwrite that
        node's message in place and stop training its generated content (demote to
        routing-only), so only the live branch trains. This only applies below
        ``fork_threshold``: a long abandoned response carries enough real signal
        to fork and train standalone.

        Forking is always safe (a rewrite mounts as routing-only); this is purely
        a cleanup. So we merge only when the mount point has exactly one assistant
        child that is a leaf, generated (``turn`` set), and short (response <
        ``fork_threshold``), and fork otherwise, since absorbing destroys a
        generated TurnRecord irreversibly.
        """
        if self._fork_threshold <= 0:
            return node, depth  # feature off
        if depth >= len(prompt_messages) or prompt_messages[depth].get("role") != "assistant":
            return node, depth  # genuine non-assistant history fork -> leave it

        asst_children = [c for c in node.children if c.role == "assistant"]
        if len(asst_children) != 1:
            if len(asst_children) > 1:
                logger.warning(
                    "record_turn(sid=%s turn=%s): %d assistant children at mount "
                    "point; can't tell which the rewrite targets, so forking.",
                    sid,
                    self._turn_count.get(sid, 0) + 1,
                    len(asst_children),
                )
            return node, depth

        rewritten_node = asst_children[0]
        if (
            rewritten_node.children
            or rewritten_node.turn is None
            or len(rewritten_node.turn.output_ids) >= self._fork_threshold
        ):
            return node, depth

        rewritten_node.metadata["merged_rewrite"] = {
            "abandoned_turn_index": rewritten_node.turn_index,
            "abandoned_response_tokens": len(rewritten_node.turn.output_ids),
        }
        rewritten_node.turn = None
        rewritten_node.turn_index = None
        rewritten_node.message = prompt_messages[depth]
        return rewritten_node, depth + 1

    def _next_node_id(self, sid: str) -> int:
        """Allocate the next per-sid node_id (monotonic from 0)."""
        nid = self._node_count.get(sid, 0)
        self._node_count[sid] = nid + 1
        return nid

    def _mount_prompt_messages(
        self,
        node: MessageNode,
        remaining_messages: list[dict[str, Any]],
        *,
        sid: str,
    ) -> MessageNode:
        for m in remaining_messages:
            child = MessageNode(role=m.get("role"), message=m)
            child.node_id = self._next_node_id(sid)
            node = node.add_child(child)
        return node

    def _attach_assistant_leaf(
        self,
        sid: str,
        node: MessageNode,
        *,
        turn: TurnRecord,
        response_message: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        asst = MessageNode(
            role="assistant",
            message=response_message,
            metadata=dict(metadata or {}),
        )
        asst.node_id = self._next_node_id(sid)
        asst.turn = turn
        asst.turn_index = self._turn_count.get(sid, 0) + 1
        node.add_child(asst)
        self._turn_count[sid] = asst.turn_index

    def _split_chain_into_builders(self, chain: list[MessageNode]) -> list[_SampleBuilder]:
        """Pack the chain's generated turns into per-Sample token builders.

        Turns flow into the current builder until one can't extend it as an
        exact prefix (re-tokenization drift past what we can drop); that turn
        opens a new builder -- a fork. A generated turn shared by sibling leaves
        is trained only on the first leaf to claim it; later leaves re-emit it
        as loss_mask=0 context so the shared prefix isn't double-counted.
        """
        asst_nodes = [n for n in chain if n.role == "assistant" and n.turn is not None]

        builders: list[_SampleBuilder] = []
        for asst_node in asst_nodes:
            trained = not asst_node.response_trained
            asst_node.response_trained = True

            if not builders or (kind := builders[-1].classify_token_drift(asst_node.turn)) is DriftKind.FORK:
                builders.append(_SampleBuilder(self._fork_threshold))
                builders[-1].append_turn(asst_node.turn, DriftKind.CLEAN, trained=trained)
            else:
                builders[-1].append_turn(asst_node.turn, kind, trained=trained)
            builders[-1].add_node(asst_node)
        return builders

    def _chain_to_samples(
        self,
        chain: list[MessageNode],
        *,
        base_sample: Sample,
        extra_metadata: dict[str, Any] | None,
    ) -> list[Sample]:
        return [
            builder.to_sample(base_sample, extra_metadata)
            for builder in self._split_chain_into_builders(chain)
            if builder.has_trained_response()
        ]

    # -------------------- identity (segment classification) ----------------

    def _classify_segments(self, root: MessageNode) -> None:
        """Stamp each generated assistant node with its identity.

        Identity has three orthogonal facets, each from a distinct content signal
        (the manager never sees SDK markers like ``parent_tool_use_id`` -- only the
        translated prompt messages), resolved per generated node:

        * **origin** (``main`` / ``sub_agent``) -- from the SYSTEM prompt. The main
          agent keeps one system prompt for the whole run; a sub-agent re-roots with
          its own (e.g. an Explore specialist prompt). A node whose leading system
          content differs from the main agent's (the earliest generated turn's) is
          a sub-agent. This is the only origin signal that survives a compaction,
          which severs the token path back to the sub-agent's first turn.
        * **is_compact_start** -- True iff a user message opening this node's turn
          starts with ``COMPACT_SUMMARY_PREFIX`` (context ran out, history replaced
          by a summary). Per-node, not propagated: only the genuine post-compact
          restart turn is a compact start, not later turns.
        * **caller_node_ids** / **match_kind** -- for sub-agents, the node(s) whose
          ``Agent``/``Task`` tool call spawned this run, matched by prompt text. The
          spawning turn's first user message equals the agent-call prompt; later
          turns inherit the caller from their nearest generated ancestor (so a
          drift-fork mid-run keeps the link). A compaction breaks that path, so a
          post-compact sub-agent turn keeps ``origin=sub_agent`` but loses the
          caller (empty list).

        Pure annotation written under ``metadata['identity_segment']``; the routing
        tree is untouched. Nodes are visited parent-before-child (pre-order) so
        caller inheritance can read an already-resolved ancestor.
        """
        gens = self._generated_nodes(root)  # pre-order: parents before children
        if not gens:
            return
        prompt_index = self._build_agent_prompt_index(gens)
        main_system = self._system_content(min(gens, key=lambda n: n.turn_index or 0))

        for gen in gens:
            origin = "sub_agent" if self._system_content(gen) != main_system else "main"
            is_compact_start = False
            own_callers: list[int] | None = None
            own_kind: str | None = None
            for text in self._lead_in_user_texts(gen):
                if text.startswith(COMPACT_SUMMARY_PREFIX):
                    is_compact_start = True
                    continue
                callers, kind = self._match_agent_prompt(text, prompt_index)
                if callers is not None:
                    own_callers, own_kind = callers, kind

            caller_node_ids: list[int] = []
            match_kind: str | None = None
            if origin == "sub_agent":
                if own_callers is not None:
                    caller_node_ids, match_kind = own_callers, own_kind
                else:  # inherit the caller from the nearest generated ancestor
                    parent_seg = self._parent_gen_segment(gen)
                    if parent_seg and parent_seg.get("origin") == "sub_agent":
                        caller_node_ids = list(parent_seg.get("caller_node_ids", []))
                        match_kind = parent_seg.get("match_kind")

            gen.metadata["identity_segment"] = {
                "origin": origin,
                "is_compact_start": is_compact_start,
                "caller_node_ids": caller_node_ids,
                "match_kind": match_kind,
            }

    @staticmethod
    def _generated_nodes(root: MessageNode) -> list[MessageNode]:
        """All generated assistant nodes (turn set) under root, pre-order."""
        out: list[MessageNode] = []
        stack = list(reversed(root.children))
        while stack:
            n = stack.pop()
            if n.role == "assistant" and n.turn is not None:
                out.append(n)
            stack.extend(reversed(n.children))
        return out

    @staticmethod
    def _system_content(node: MessageNode) -> str:
        """Leading system-message content on ``node``'s path (``""`` if none)."""
        for n in node.path_from_root():
            if n.role == "system":
                content = (n.message or {}).get("content")
                return content if isinstance(content, str) else ""
        return ""

    @staticmethod
    def _parent_gen_segment(gen: MessageNode) -> dict[str, Any] | None:
        """``identity_segment`` of the nearest generated ancestor, if resolved."""
        node = gen.parent
        while node is not None and not node.is_root:
            if node.role == "assistant" and node.turn is not None:
                return node.metadata.get("identity_segment")
            node = node.parent
        return None

    def _build_agent_prompt_index(self, gens: list[MessageNode]) -> dict[str, list[int]]:
        """Map each prior agent-call prompt text -> the node_ids that issued it.

        Scans every generated assistant node's ``message['tool_calls']`` for
        ``Agent``/``Task`` calls and indexes their ``arguments['prompt']``. A
        prompt may map to several caller node_ids (parallel fan-out reuses the
        same prompt, or one node issues several agent calls), so the value is a
        list -- the full candidate set, deduped, in node_id order.
        """
        index: dict[str, list[int]] = {}
        for node in gens:
            msg = node.message or {}
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") if isinstance(tc, dict) else None
                if not isinstance(fn, dict) or fn.get("name") not in SUBAGENT_TOOL_NAMES:
                    continue
                args = fn.get("arguments")
                prompt = args.get("prompt") if isinstance(args, dict) else None
                if not isinstance(prompt, str) or not prompt:
                    continue
                callers = index.setdefault(prompt, [])
                if node.node_id not in callers:
                    callers.append(node.node_id)
        return index

    @staticmethod
    def _lead_in_user_texts(gen: MessageNode) -> list[str]:
        """User-message texts between ``gen`` and the previous generated node.

        These routing-only user nodes are what opened ``gen``'s turn -- the
        compaction summary or the sub-agent task prompt land here. Walks up from
        ``gen`` collecting user contents until it hits another generated assistant
        (the previous turn's tail) or the root.
        """
        texts: list[str] = []
        node = gen.parent
        while node is not None and not node.is_root:
            if node.role == "assistant" and node.turn is not None:
                break
            if node.role == "user":
                content = (node.message or {}).get("content")
                if isinstance(content, str) and content:
                    texts.append(content)
            node = node.parent
        return texts

    @staticmethod
    def _match_agent_prompt(text: str, prompt_index: dict[str, list[int]]) -> tuple[list[int] | None, str | None]:
        """Match a lead-in user text against the agent-call prompt index.

        Exact (byte-equal) match wins and is reported as ``"exact"``. Failing
        that, a relaxed pass tolerates whitespace drift and prefix wrapping (cc
        prepends a ``<system-reminder>`` block or trims trailing space): the
        index prompt is accepted if it equals the text after stripping, or the
        text starts with / contains the stripped prompt. Relaxed hits report
        ``"approx"``. Returns ``(None, None)`` when nothing matches.
        """
        if text in prompt_index:
            return list(prompt_index[text]), "exact"
        stripped = text.strip()
        for prompt, callers in prompt_index.items():
            p = prompt.strip()
            if not p:
                continue
            if stripped == p or stripped.startswith(p) or p in stripped:
                return list(callers), "approx"
        return None, None


__all__ = [
    "TrajectoryManager",
    "TurnRecord",
]
