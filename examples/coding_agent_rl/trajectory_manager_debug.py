"""Debug-only tree dumper for src_v2.trajectory_manager.

This module is OPTIONAL. trajectory_manager.py itself does not import it.
It walks the public Node attributes (role / messages / children / turn_*)
of a TrajectoryManager session and renders an ASCII tree for human reading
or a structured dict for JSON dumps.

Kept separate from the core so the delivered trajectory_manager.py can stay
minimal (router + linearizer only, ~430 lines).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def dump_tree_txt(
    manager,
    sid: str,
    *,
    max_text_chars: int = 80,
    show_tokens: bool = False,
) -> str:
    """Render manager._trees[sid] as ASCII text.

    Returns ``"<no session: {sid}>"`` if the sid has no tree (drained or
    never opened). Otherwise emits a header line plus one indented row per
    non-root node, listing role / msg count / turn snapshot fields for
    assistant leaves.
    """
    root = manager._trees.get(sid)
    if root is None:
        return f"<no session: {sid}>"
    non_root_count = sum(1 for _ in _iter_non_root(root))
    leaf_count = sum(1 for leaf in root.leaves() if not leaf.is_root)
    lines: list[str] = [
        f"session={sid} turns={manager.turn_count(sid)} leaves={leaf_count} nodes={non_root_count}",
        "root",
    ]
    _render_subtree(
        root,
        lines,
        depth=0,
        max_text_chars=max_text_chars,
        show_tokens=show_tokens,
    )
    return "\n".join(lines)


def dump_tree_json(manager, sid: str, *, include_messages: bool = False) -> dict[str, Any]:
    """Render manager._trees[sid] as a JSON-serializable dict.

    Returns ``{"sid": sid, "found": False}`` if the sid has no tree.

    Pass ``include_messages=True`` to embed each node's raw ``messages`` list
    (preserves ``reasoning_content`` and thinking blocks for downstream
    inspection). Default stays off because messages can be very large.
    """
    root = manager._trees.get(sid)
    if root is None:
        return {"sid": sid, "found": False}
    non_root_count = sum(1 for _ in _iter_non_root(root))
    leaf_count = sum(1 for leaf in root.leaves() if not leaf.is_root)
    return {
        "sid": sid,
        "found": True,
        "turns": manager.turn_count(sid),
        "leaves": leaf_count,
        "nodes_total": non_root_count,
        "root": _node_to_json(root, include_messages=include_messages),
    }


# ---------------------------------------------------------------------------
# private helpers (moved from trajectory_manager.py)
# ---------------------------------------------------------------------------


def _text_summary(messages: list[dict[str, Any]]) -> str:
    """Concatenate content from a list of OpenAI-shape messages for display.

    Picks up three sources per message:
      * top-level ``reasoning_content`` (OpenAI/sglang shape — sibling to ``content``)
      * ``content`` blocks of type ``text`` (or a bare string ``content``)
      * ``content`` blocks of type ``thinking`` (Anthropic wire shape — see
        adapters/anthropic.py _build_blocks_and_response_message)
    Reasoning/thinking is emitted as ``reason:<...>`` so the segment is visible
    in the txt dump and unambiguous vs the visible-text segment.
    """
    out: list[str] = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content")
        thinkings: list[str] = []
        reasoning = m.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            thinkings.append(reasoning)
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for b in content:
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                if bt == "text":
                    parts.append(str(b.get("text", "")))
                elif bt == "thinking":
                    thinkings.append(str(b.get("thinking", "")))
            text = "".join(parts)
        elif content is None:
            text = ""
        else:
            text = str(content)
        segs: list[str] = []
        if thinkings:
            segs.append("reason:" + "".join(thinkings))
        if text:
            segs.append(text)
        out.append(f"{role}:" + " ".join(segs))
    return " | ".join(out)


def _iter_non_root(root) -> Iterator:
    stack: list = list(root.children)
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.children)


def _render_subtree(
    node,
    lines: list[str],
    *,
    depth: int,
    max_text_chars: int,
    show_tokens: bool,
) -> None:
    indent = "    " * depth
    for child in node.children:
        text = _text_summary(child.messages)
        if len(text) > max_text_chars:
            text = text[:max_text_chars] + "..."
        text = text.replace("'", "\\'")
        fields = [f"[{child.role}]", f"msgs={len(child.messages)}"]
        if child.role == "assistant" and child.turn_index is not None:
            tp = len(child.turn_prompt_ids or [])
            tr = len(child.turn_response_ids or [])
            fields.append(f"turn={child.turn_index}")
            fields.append(f"prompt_ids=({tp})")
            fields.append(f"response_ids=({tr})")
            fields.append(f"finish={child.turn_finish_reason}")
            fields.append(f"has_logprobs={child.turn_response_logprobs is not None}")
        fields.append(f"text='{text}'")
        lines.append(f"{indent}└── " + " ".join(fields))
        if show_tokens and child.turn_response_ids:
            ids = child.turn_response_ids
            if len(ids) > 40:
                head = ", ".join(str(x) for x in ids[:32])
                tail = ", ".join(str(x) for x in ids[-8:])
                rendered = f"[{head}, ..., {tail}]"
            else:
                rendered = "[" + ", ".join(str(x) for x in ids) + "]"
            lines.append(f"{indent}        response_ids: {rendered}")
        _render_subtree(
            child,
            lines,
            depth=depth + 1,
            max_text_chars=max_text_chars,
            show_tokens=show_tokens,
        )


def _node_to_json(node, *, include_messages: bool = False) -> dict[str, Any]:
    d: dict[str, Any] = {
        "role": node.role,
        "msg_count": len(node.messages),
        "text": _text_summary(node.messages),
        "children": [_node_to_json(c, include_messages=include_messages) for c in node.children],
    }
    if include_messages:
        d["messages"] = node.messages
    if node.role == "assistant" and node.turn_index is not None:
        d["turn_index"] = node.turn_index
        d["turn_prompt_ids_len"] = len(node.turn_prompt_ids or [])
        d["turn_response_ids_len"] = len(node.turn_response_ids or [])
        d["finish_reason"] = node.turn_finish_reason
        d["has_logprobs"] = node.turn_response_logprobs is not None
    return d


__all__ = [
    "dump_tree_txt",
    "dump_tree_json",
]
