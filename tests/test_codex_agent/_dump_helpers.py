"""Debug + dump helpers for the codex-agent e2e test.

Mirror of ``tests/test_coding_agent/_dump_helpers.py`` but tuned for the
OpenAI Chat-Completions wire used by the Codex CLI:

  * middleware captures ``/v1/chat/completions`` (NOT ``/v1/messages``)
  * SSE decoder walks chat-completions chunks (``data: {chatcmpl ...}``)
    instead of Anthropic's ``event: content_block_delta`` shape
  * the adapter target is :class:`OpenAIAdapter`; ``Debug`` and the dump
    layout are otherwise identical

The dump layout (``turn_NNNN_*.json``, ``trajectory_tree.{txt,json}``,
``trajectory.json``) matches the anthropic side so cross-adapter comparisons
stay straightforward.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary

logger = logging.getLogger("test_codex_dump_helpers")


def dump_tree_txt(
    manager,
    sid: str,
    *,
    max_text_chars: int = 80,
    show_tokens: bool = False,
) -> str:
    """Render ``manager._trees[sid]`` as ASCII text."""
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
    """Render ``manager._trees[sid]`` as a JSON-serializable dict."""
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
# private helpers
# ---------------------------------------------------------------------------


def _text_summary(messages: list[dict[str, Any]]) -> str:
    """Concatenate display text from a list of OpenAI-shape messages."""
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
    "install_dump_layer",
    "Debug",
]


# ===========================================================================
# Dump-layer factory + Debug handle (debug-only; used only by codex e2e test).
# ===========================================================================


def install_dump_layer(
    adapter,
    *,
    run_dir,
    tokenizer,
    sid_dump_dir: dict[str, str] | None = None,
) -> Debug:
    """Install dump middleware on the OpenAI adapter app and wire the per-turn hook."""
    handle = Debug(Path(run_dir), tokenizer=tokenizer, sid_dump_dir=sid_dump_dir)
    adapter.app.middlewares.append(build_dump_middleware(handle))
    adapter.on_turn_appended = handle.on_turn_appended_bridge
    return handle


class Debug:
    """Per-turn / per-session disk dump hooks (OpenAI flavor)."""

    def __init__(
        self,
        run_dir: Path,
        *,
        tokenizer,
        sid_dump_dir: dict[str, str] | None = None,
    ) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self.tok = tokenizer
        self.sid_dump_dir = sid_dump_dir
        self._last_turn: dict[str, int] = {}

    def _sid_dir(self, sid: str) -> Path:
        if self.sid_dump_dir is not None:
            override = self.sid_dump_dir.get(sid)
            if override:
                d = Path(override)
                d.mkdir(parents=True, exist_ok=True)
                return d
        safe = sid.replace("/", "_").replace("..", "_")[:120]
        d = self.run_dir / safe
        d.mkdir(parents=True, exist_ok=True)
        return d

    def on_request(self, sid: str, n: int, body_obj, body_bytes: bytes) -> None:
        d = self._sid_dir(sid)
        if body_obj is not None:
            _dump_json(d / f"turn_{n:04d}_request.json", body_obj)
        else:
            (d / f"turn_{n:04d}_request.raw").write_bytes(body_bytes)

    def on_sse(self, sid: str, n: int, sse_bytes: bytes, resp) -> None:
        from aiohttp import web as _web

        d = self._sid_dir(sid)
        if sse_bytes:
            sse_path = d / f"turn_{n:04d}_response.sse"
            decoded = _decoded_text_from_sse(sse_bytes)
            tail = (
                b"\n# ----------------------------------------------------------\n"
                b"# decoded assistant output (text/reasoning/tool_call reconstructed\n"
                b"# from chat.completion.chunk frames above; not part of the wire):\n"
                b"# ----------------------------------------------------------\n" + decoded.encode("utf-8") + b"\n"
            )
            sse_path.write_bytes(sse_bytes + tail)
            return
        if isinstance(resp, _web.Response) and resp.body is not None:
            body = resp.body
            if isinstance(body, (bytes, bytearray)):
                try:
                    _dump_json(
                        d / f"turn_{n:04d}_response.json",
                        json.loads(bytes(body)),
                    )
                except (json.JSONDecodeError, UnicodeDecodeError):
                    (d / f"turn_{n:04d}_response.raw").write_bytes(bytes(body))
            else:
                (d / f"turn_{n:04d}_response.bin").write_bytes(bytes(body))

    def on_sglang_pair(
        self,
        sid: str,
        n: int,
        sglang_req: dict | None,
        sglang_resp: dict | None,
        *,
        req_bytes: bytes | None = None,
        resp_bytes: bytes | None = None,
    ) -> None:
        d = self._sid_dir(sid)
        if sglang_req is not None:
            _dump_json(d / f"turn_{n:04d}_sglang_request.json", sglang_req)
        elif req_bytes is not None:
            (d / f"turn_{n:04d}_sglang_request.raw").write_bytes(req_bytes)
        if sglang_resp is not None:
            _dump_json(d / f"turn_{n:04d}_sglang_response.json", sglang_resp)
        elif resp_bytes is not None:
            (d / f"turn_{n:04d}_sglang_response.raw").write_bytes(resp_bytes)

    def on_turn_appended(
        self,
        sid: str,
        n: int,
        prompt_messages,
        tools,
        response_message,
        prompt_ids,
        response_ids,
        finish_reason,
    ) -> None:
        try:
            _dump_json(
                self._sid_dir(sid) / f"turn_{n:04d}_openai.json",
                {
                    "sid": sid,
                    "turn": n,
                    "prompt_messages": prompt_messages,
                    "tools": tools,
                    "response_message": response_message,
                    "prompt_ids_len": len(prompt_ids),
                    "response_ids_len": len(response_ids),
                    "finish_reason": finish_reason,
                },
            )
        except Exception:
            logger.exception("turn %d: openai dump failed", n)

    def on_turn_appended_bridge(
        self,
        sid,
        prompt_messages,
        tools,
        response_message,
        prompt_ids,
        response_ids,
        finish_reason,
    ) -> None:
        n = self._last_turn.get(sid, 0)
        self.on_turn_appended(
            sid,
            n,
            prompt_messages,
            tools,
            response_message,
            prompt_ids,
            response_ids,
            finish_reason,
        )

    def on_drain_start(self, sid: str, manager) -> None:
        d = self._sid_dir(sid)
        try:
            (d / "trajectory_tree.txt").write_text(
                dump_tree_txt(manager, sid),
                encoding="utf-8",
            )
            _dump_json(
                d / "trajectory_tree.json",
                dump_tree_json(manager, sid, include_messages=True),
            )
        except Exception:
            logger.exception("dump_tree(%s) failed", sid)

    def on_drain_done(self, sid: str, samples) -> None:
        d = self._sid_dir(sid)
        try:
            rendered = [sample_dict_with_text(s, tokenizer=self.tok) for s in samples]
            _dump_json(d / "trajectory.json", rendered)
        except Exception:
            logger.exception("trajectory.json dump for sid=%s failed", sid)


# ===========================================================================
# aiohttp dump middleware (per-turn capture for /v1/chat/completions)
# ===========================================================================


def build_dump_middleware(debug: Debug):
    """Build an aiohttp middleware that captures /v1/chat/completions turns."""
    import asyncio as _asyncio
    import itertools as _itertools

    from aiohttp import web as _web

    turn_counters: dict[str, _itertools.count] = {}
    counter_lock = _asyncio.Lock()

    async def _next_turn(sid: str) -> int:
        async with counter_lock:
            ctr = turn_counters.get(sid)
            if ctr is None:
                ctr = _itertools.count(1)
                turn_counters[sid] = ctr
            return next(ctr)

    def _sid_of(request) -> str:
        # OpenAI clients (Codex CLI) authenticate via `Authorization: Bearer
        # <sid>`. Fallback to X-Api-Key kept for symmetry with the anthropic
        # middleware so the dump layer can be reused with a mixed fleet.
        auth = request.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            sid = auth[7:].strip()
            if sid:
                return sid
        api_key = request.headers.get("X-Api-Key", "")
        if api_key:
            return api_key.strip()
        return "default"

    @_web.middleware
    async def dump_mw(request, handler):
        if request.path != "/v1/chat/completions":
            return await handler(request)

        sid = _sid_of(request)
        n = await _next_turn(sid)
        debug._last_turn[sid] = n

        body_bytes = await request.read()
        body_obj: dict | None
        try:
            body_obj = json.loads(body_bytes)
        except (json.JSONDecodeError, TypeError):
            body_obj = None

        debug.on_request(sid, n, body_obj, body_bytes)

        captured: list[bytes] = []
        task = _asyncio.current_task()
        if task is not None:
            _SSE_CAPTURE[task] = captured
        try:
            resp = await handler(request)
        finally:
            if task is not None:
                _SSE_CAPTURE.pop(task, None)

        sse_bytes = b"".join(captured) if captured else b""
        debug.on_sse(sid, n, sse_bytes, resp)
        return resp

    return dump_mw


def _install_sse_capture_patch():
    """Patch ``web.StreamResponse.write`` once per process."""
    import asyncio as _asyncio

    from aiohttp import web as _web

    global _orig_stream_write, _SSE_CAPTURE_PATCHED
    if _SSE_CAPTURE_PATCHED:
        return
    _orig_stream_write = _web.StreamResponse.write

    async def _capturing_stream_write(self, data):  # type: ignore[no-untyped-def]
        cap = _SSE_CAPTURE.get(_asyncio.current_task())
        if cap is not None:
            cap.append(bytes(data))
        return await _orig_stream_write(self, data)

    _web.StreamResponse.write = _capturing_stream_write  # type: ignore[assignment]
    _SSE_CAPTURE_PATCHED = True


_SSE_CAPTURE: WeakKeyDictionary[Any, list[bytes]] = WeakKeyDictionary()
_orig_stream_write = None
_SSE_CAPTURE_PATCHED = False
_install_sse_capture_patch()


# ===========================================================================
# Local module helpers
# ===========================================================================


def _dump_json(path: Path, obj) -> None:
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _decoded_text_from_sse(sse_bytes: bytes) -> str:
    """Reconstruct the assistant's visible output from a captured chat-completions SSE stream.

    Each frame is shaped::

        data: {"id": ..., "object": "chat.completion.chunk",
               "choices": [{"index": 0, "delta": {...}, "finish_reason": ...}]}

    Deltas accumulate into ``content``, ``reasoning_content``, and
    per-tool-call function name + arguments. Terminated by ``data: [DONE]``.
    """
    content_buf: list[str] = []
    reasoning_buf: list[str] = []
    # tool_calls keyed by chunk-reported index; each value carries the
    # cumulative name + argument fragments.
    tool_calls: dict[int, dict[str, Any]] = {}

    for line in sse_bytes.decode("utf-8", errors="replace").splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str) and content:
            content_buf.append(content)
        reasoning = delta.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            reasoning_buf.append(reasoning)
        for tc in delta.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            idx = tc.get("index", 0)
            slot = tool_calls.setdefault(
                idx,
                {"id": "", "name": "", "arguments": ""},
            )
            if tc.get("id"):
                slot["id"] = tc["id"]
            function = tc.get("function") or {}
            if function.get("name"):
                slot["name"] = function["name"]
            args_chunk = function.get("arguments")
            if isinstance(args_chunk, str):
                slot["arguments"] += args_chunk

    parts: list[str] = []
    if reasoning_buf:
        parts.append("[reasoning]\n" + "".join(reasoning_buf))
    if content_buf:
        parts.append("[content]\n" + "".join(content_buf))
    for idx in sorted(tool_calls):
        tc = tool_calls[idx]
        parts.append(f"[tool_call name={tc['name']} id={tc['id']}]\n{tc['arguments']}")
    return "\n\n".join(parts)


def sample_dict_with_text(sample, *, tokenizer) -> dict[str, Any]:
    """Render a slime Sample as a JSON-serializable dict with decoded text."""
    data = sample.to_dict() if hasattr(sample, "to_dict") else dict(sample.__dict__)
    tokens = data.get("tokens") or []
    rlen = int(data.get("response_length") or 0)
    prompt_ids = tokens[: len(tokens) - rlen]
    response_ids = tokens[len(tokens) - rlen :]
    if tokenizer is not None:
        try:
            data["prompt_text"] = tokenizer.decode(prompt_ids, skip_special_tokens=False) if prompt_ids else ""
        except Exception as e:
            data["prompt_text"] = f"<decode failed: {e!r}>"
        try:
            data["response_text"] = tokenizer.decode(response_ids, skip_special_tokens=False) if response_ids else ""
        except Exception as e:
            data["response_text"] = f"<decode failed: {e!r}>"
    return data
