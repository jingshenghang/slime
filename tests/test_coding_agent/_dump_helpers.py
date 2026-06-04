"""Debug + dump helpers for the coding-agent e2e test.

This module hosts every piece of debug-only machinery the e2e test needs to
turn an in-process AnthropicAdapter session into the same on-disk layout that
``examples/coding_agent_rl/launch_swe.py`` historically produced:

  * tree dumpers (``dump_tree_txt`` / ``dump_tree_json``) used by the
    drain-time snapshot writer and by the trajectory_manager debug pytest
  * ``install_dump_layer`` + ``Debug``: factory and per-turn hook surface
    that mounts an aiohttp dump middleware on the adapter app and
    captures request / response / sglang wire data per (sid, turn).

Keeping all of this under ``tests/test_coding_agent/`` keeps debug surface out
of the runtime ``slime/agent/`` tree, per the project constraint that the
runtime-only delivery is ``slime/agent/trajectory_manager.py`` plus tests.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary

logger = logging.getLogger("test_dump_helpers")


def dump_tree_txt(
    manager,
    sid: str,
    *,
    max_text_chars: int = 80,
    show_tokens: bool = False,
) -> str:
    """Render ``manager._trees[sid]`` as ASCII text.

    Returns ``"<no session: {sid}>"`` when the sid has no tree (already
    drained or never opened). Otherwise emits a header line plus one indented
    row per non-root node, listing role / msg count / turn snapshot fields
    for assistant leaves.
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
    """Render ``manager._trees[sid]`` as a JSON-serializable dict.

    Returns ``{"sid": sid, "found": False}`` when the sid has no tree.

    ``include_messages=True`` embeds each node's raw ``messages`` list
    (preserves ``reasoning_content`` and thinking blocks for downstream
    inspection). Default off because messages can be very large.
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
# private helpers
# ---------------------------------------------------------------------------


def _text_summary(messages: list[dict[str, Any]]) -> str:
    """Concatenate display text from a list of OpenAI-shape messages.

    Picks up three sources per message:
      * top-level ``reasoning_content`` (OpenAI/sglang shape — sibling to
        ``content``)
      * ``content`` blocks of type ``text`` (or a bare string ``content``)
      * ``content`` blocks of type ``thinking`` (Anthropic wire shape — see
        ``adapters/anthropic.py _build_blocks_and_response_message``)

    Reasoning/thinking is emitted as ``reason:<...>`` so the segment is
    visible in the txt dump and unambiguous vs the visible-text segment.
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
    "install_dump_layer",
    "Debug",
]


# ===========================================================================
# Dump-layer factory + Debug handle (debug-only; used only by SWE e2e test).
# ===========================================================================


def install_dump_layer(
    adapter,
    *,
    run_dir,
    tokenizer,
    sid_dump_dir: dict[str, str] | None = None,
) -> Debug:
    """One-shot installer: create a ``Debug`` handle, register the dump
    middleware on the adapter app, and wire the adapter-side per-turn hook.

    Parameters
    ----------
    adapter : AnthropicAdapter
        Target adapter; ``adapter.app.middlewares`` is appended in place and
        ``adapter.on_turn_appended`` is set to the Debug bridge.
    run_dir : str | Path
        Filesystem directory that holds per-sid dump folders. Created if
        missing.
    tokenizer
        Tokenizer used by ``Debug.on_drain_done`` to decode
        ``prompt_text`` / ``response_text`` for trajectory.json.
    sid_dump_dir : dict[str, str] | None
        Optional per-sid override: when ``sid_dump_dir[sid]`` is set, all
        artifacts for that sid land there directly (instead of
        ``run_dir/<sid>/``). The dict can be mutated after install — the
        Debug handle reads it on every hook call.
    """
    handle = Debug(Path(run_dir), tokenizer=tokenizer, sid_dump_dir=sid_dump_dir)
    adapter.app.middlewares.append(build_dump_middleware(handle))
    adapter.on_turn_appended = handle.on_turn_appended_bridge
    return handle


class Debug:
    """Per-turn / per-session disk dump hooks.

    Each hook writes one or two files into the sid's dump directory. The
    directory is resolved per-call via ``sid_dump_dir[sid]`` if present,
    otherwise ``run_dir/<sid_safe>/`` where slashes are sanitized.
    """

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
        # Per-sid turn counter populated by the dump middleware so the
        # adapter-side ``on_turn_appended`` bridge can look up the current
        # turn number for openai.json filenames. Plain dict (no lock) is
        # fine: the middleware writes always happen on the SAME asyncio
        # task as the adapter's append_turn call for that request.
        self._last_turn: dict[str, int] = {}

    # ---- sid -> dir resolution ----------------------------------------- #

    def _sid_dir(self, sid: str) -> Path:
        if self.sid_dump_dir is not None:
            override = self.sid_dump_dir.get(sid)
            if override:
                d = Path(override)
                d.mkdir(parents=True, exist_ok=True)
                return d
        # Slashes (cc subagent ids contain them sometimes) would escape run_dir.
        safe = sid.replace("/", "_").replace("..", "_")[:120]
        d = self.run_dir / safe
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---- hooks --------------------------------------------------------- #

    def on_request(self, sid: str, n: int, body_obj, body_bytes: bytes) -> None:
        """Persist raw incoming /v1/messages request to disk."""
        d = self._sid_dir(sid)
        if body_obj is not None:
            _dump_json(d / f"turn_{n:04d}_request.json", body_obj)
        else:
            (d / f"turn_{n:04d}_request.raw").write_bytes(body_bytes)

    def on_request_scrubbed(self, sid: str, n: int, body_obj) -> None:
        """Persist body after billing-header scrub / system-fold mutations.

        Only called when the in-middleware adapter scrubbing actually
        changed the body.
        """
        _dump_json(
            self._sid_dir(sid) / f"turn_{n:04d}_request_scrubbed.json",
            body_obj,
        )

    def on_sse(self, sid: str, n: int, sse_bytes: bytes, resp) -> None:
        """Persist the outgoing response.

        For an SSE stream: write the raw frame bytes to ``.sse`` and append
        a decoded human-readable tail. For a non-stream JSON response,
        try to decode the body and write to ``.json`` (falling back to
        ``.raw`` if it isn't valid JSON, or ``.bin`` for non-bytes bodies).
        """
        from aiohttp import web as _web

        d = self._sid_dir(sid)
        if sse_bytes:
            sse_path = d / f"turn_{n:04d}_response.sse"
            decoded = _decoded_text_from_sse(sse_bytes)
            tail = (
                b"\n# ----------------------------------------------------------\n"
                b"# decoded assistant output (text/thinking/tool_use reconstructed\n"
                b"# from SSE frames above; not part of the wire protocol):\n"
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
        """Persist the upstream sglang request/response pair for this turn.

        Either side may be ``None`` (e.g. when only one half is captured),
        in which case the corresponding file is skipped. ``req_bytes`` /
        ``resp_bytes`` provide a fallback when the structured dict isn't
        available — we then write a ``.raw`` instead of ``.json``.
        """
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
        """Persist the OpenAI-shape view of an append_turn call.

        Captures everything the manager saw, plus prompt/response id
        lengths. Wrapped in try/except so a dump failure can never break
        the SSE response.
        """
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
        """Adapter-side ``on_turn_appended`` bridge.

        The adapter hook signature has no turn number; we look it up via
        the per-sid counter populated by the dump middleware.
        """
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
        """Snapshot the live tree to ``trajectory_tree.{txt,json}``.

        Must be called BEFORE ``manager.get_trajectory(sid, drop=True)``
        because the drain pops the per-sid tree.
        """
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
        """Persist the drained samples to ``trajectory.json`` with
        decoded ``prompt_text`` / ``response_text`` per sample.

        ``trajectory.json`` is the debug-only enriched payload; the over
        the wire ``/get_trajectory`` response intentionally does NOT
        carry the decoded text fields.
        """
        d = self._sid_dir(sid)
        try:
            rendered = [sample_dict_with_text(s, tokenizer=self.tok) for s in samples]
            _dump_json(d / "trajectory.json", rendered)
        except Exception:
            logger.exception("trajectory.json dump for sid=%s failed", sid)


# ===========================================================================
# aiohttp dump middleware (per-turn capture for /v1/messages)
# ===========================================================================


def build_dump_middleware(debug: Debug):
    """Build an aiohttp middleware that captures /v1/messages turns.

    The middleware:
      * derives the sid from Authorization / X-Api-Key (mirrors the adapter
        helper);
      * allocates a per-sid monotonic turn number;
      * captures the request body (for ``on_request``);
      * captures the SSE response stream via a module-level
        ``WeakKeyDictionary`` keyed on ``asyncio.Task`` (avoids the
        per-task monkey-patch race described in
        ``claude-tooling/aiohttp_per_task_monkeypatch_race``);
      * invokes ``on_sse`` after the handler returns.

    Other paths (``/healthz`` / ``/v1/models`` / ``/get_trajectory``) pass
    through with zero overhead.
    """
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
        if request.path != "/v1/messages":
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

        # Per-task SSE buffer registered BEFORE the handler runs.
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


# ---------------------------------------------------------------------------
# SSE capture: module-level WeakKeyDictionary keyed by asyncio.Task.
#
# Rationale: aiohttp serves multiple concurrent requests on a single event
# loop; if we monkey-patch ``StreamResponse.write`` to feed an instance-
# scoped buffer we get cross-request bleed under concurrency. Keying by
# the *current Task* localises captures to the request that owns the
# coroutine. Untracked tasks pass through with zero overhead.
#
# WeakKeyDictionary lets entries vanish if a Task is GC'd before we
# explicitly pop them; the explicit ``pop`` in the middleware ``finally``
# block remains the primary cleanup, this is just a safety net.
# See memory ``claude-tooling/aiohttp_per_task_monkeypatch_race`` for
# the original incident report.
# ---------------------------------------------------------------------------


def _install_sse_capture_patch():
    """Patch ``web.StreamResponse.write`` once per process.

    Idempotent: subsequent calls are no-ops. The wrapper consults the
    module-level ``_SSE_CAPTURE`` dict for the *current* task; if the
    task is registered, the written chunk is mirrored into the captor's
    list before delegating to the original ``write``.
    """
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
    """Reconstruct the assistant's visible output from a captured SSE stream.

    Human-readable tail only — the canonical wire data is the raw SSE
    frames above this rendering inside ``turn_NNNN_response.sse``.
    """
    parts: list[str] = []
    text_buf: dict[int, str] = {}
    think_buf: dict[int, str] = {}
    tool_calls: dict[int, dict] = {}

    for line in sse_bytes.decode("utf-8", errors="replace").splitlines():
        if not line.startswith("data: "):
            continue
        try:
            event = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        et = event.get("type")
        if et == "content_block_start":
            idx = event.get("index", 0)
            cb = event.get("content_block") or {}
            if cb.get("type") == "tool_use":
                tool_calls[idx] = {
                    "name": cb.get("name", ""),
                    "id": cb.get("id", ""),
                    "input_chunks": [],
                }
        elif et == "content_block_delta":
            idx = event.get("index", 0)
            delta = event.get("delta") or {}
            dt = delta.get("type")
            if dt == "text_delta":
                text_buf[idx] = text_buf.get(idx, "") + delta.get("text", "")
            elif dt == "thinking_delta":
                think_buf[idx] = think_buf.get(idx, "") + delta.get("thinking", "")
            elif dt == "input_json_delta":
                tc = tool_calls.setdefault(idx, {"name": "?", "id": "", "input_chunks": []})
                tc["input_chunks"].append(delta.get("partial_json", ""))

    if think_buf:
        parts.append("[thinking]\n" + "".join(think_buf[i] for i in sorted(think_buf)))
    if text_buf:
        parts.append("[text]\n" + "".join(text_buf[i] for i in sorted(text_buf)))
    for idx in sorted(tool_calls):
        tc = tool_calls[idx]
        input_str = "".join(tc["input_chunks"])
        parts.append(f"[tool_use name={tc['name']} id={tc['id']}]\n{input_str}")
    return "\n\n".join(parts)


def sample_dict_with_text(sample, *, tokenizer) -> dict[str, Any]:
    """Render a slime Sample as a JSON-serializable dict with
    ``prompt_text`` / ``response_text`` decoded via tokenizer.

    Used only by the debug dumper for ``trajectory.json`` on disk. The
    HTTP ``/get_trajectory`` response intentionally does NOT include
    these fields (consumers can decode ``sample.tokens`` themselves).
    """
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
