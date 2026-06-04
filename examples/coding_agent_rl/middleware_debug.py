"""Debug-only artifact dumper for src_v2.middleware.

This module is OPTIONAL. middleware.py imports it conditionally only when
``--run-dir`` is passed on the CLI. install() returns a Debug handle that
implements 6 hook methods invoked from inside middleware.py:

  on_request(sid, n, body_obj, body_bytes)
    -> turn_NNNN_request.json (or .raw)
  on_request_scrubbed(sid, n, body_obj)
    -> turn_NNNN_request_scrubbed.json
  on_sse(sid, n, sse_bytes, resp)
    -> turn_NNNN_response.sse  (with decoded tail appended)
    -> or turn_NNNN_response.json / .raw / .bin for non-SSE responses
  on_sglang_pair(sid, n, sglang_req, sglang_resp)
    -> turn_NNNN_sglang_request.json + turn_NNNN_sglang_response.json
  on_turn_appended(sid, n, prompt_messages, tools, response_message,
                   prompt_ids, response_ids, finish_reason)
    -> turn_NNNN_openai.json
  on_drain_start(sid, manager)
    -> trajectory_tree.txt + trajectory_tree.json (called BEFORE drain)
  on_drain_done(sid, samples)
    -> trajectory.json (called AFTER drain)

trajectory.json contains debug-decorated sample dicts (with prompt_text /
response_text decoded via tokenizer). The HTTP /get_trajectory response
itself does NOT carry those fields.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from aiohttp import web

logger = logging.getLogger("middleware_debug")


def install(run_dir, tokenizer, manager, args, sid_dump_dir=None) -> Debug:
    """Build and return a Debug handle. Creates run_dir if missing and
    writes middleware_meta.json once at startup.

    ``sid_dump_dir`` (optional ``dict[str, str]``) lets the caller route a
    sid's dumps to an explicit absolute path instead of ``run_dir/<sid>/``.
    Lookup is per-sid; unmapped sids fall back to the old layout.
    """
    return Debug(Path(run_dir), tokenizer, manager, args, sid_dump_dir=sid_dump_dir)


class Debug:
    def __init__(self, run_dir: Path, tokenizer, manager, args, sid_dump_dir=None):
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self.tok = tokenizer
        self.manager = manager
        self.sid_dump_dir = sid_dump_dir
        self._stamp_meta(args)

    # ---------------- meta + sid dir ---------------- #

    def _stamp_meta(self, args) -> None:
        self_url = f"http://127.0.0.1:{args.port}"
        (self.run_dir / "middleware_meta.json").write_text(
            json.dumps(
                {
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "self_url": self_url,
                    "real_sglang_url": args.sglang_url.rstrip("/"),
                    "model": args.model,
                    "tool_parser": args.tool_parser,
                    "reasoning_parser": args.reasoning_parser,
                    "trajectory_format": ("slime.Sample[] (variable-global-batch-size, PR #1933)"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _sid_dir(self, sid: str) -> Path:
        # Per-sid override (set via /register_sid from launch_swe) lets us
        # drop the redundant mw/ layer and write turn dumps directly into
        # the instance dir alongside meta.json / stdout.log.
        if self.sid_dump_dir is not None:
            override = self.sid_dump_dir.get(sid)
            if override:
                d = Path(override)
                d.mkdir(parents=True, exist_ok=True)
                return d
        # Slashes (CC subagent ids contain them sometimes) would escape run_dir.
        safe = sid.replace("/", "_").replace("..", "_")[:120]
        d = self.run_dir / safe
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---------------- hooks ---------------- #

    def on_request(self, sid: str, n: int, body_obj, body_bytes: bytes) -> None:
        d = self._sid_dir(sid)
        if body_obj is not None:
            _dump_json(d / f"turn_{n:04d}_request.json", body_obj)
        else:
            (d / f"turn_{n:04d}_request.raw").write_bytes(body_bytes)

    def on_request_scrubbed(self, sid: str, n: int, body_obj) -> None:
        _dump_json(
            self._sid_dir(sid) / f"turn_{n:04d}_request_scrubbed.json",
            body_obj,
        )

    def on_sse(self, sid: str, n: int, sse_bytes: bytes, resp) -> None:
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
        if isinstance(resp, web.Response) and resp.body is not None:
            body = resp.body
            if isinstance(body, (bytes, bytearray)):
                try:
                    _dump_json(
                        d / f"turn_{n:04d}_response.json",
                        json.loads(bytes(body)),
                    )
                except json.JSONDecodeError:
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

    def on_drain_start(self, sid: str, manager) -> None:
        """Called BEFORE manager.get_trajectory(sid) drains the tree.

        Dumps the live tree to trajectory_tree.txt / trajectory_tree.json.
        Must come before drain because get_trajectory(drop=True) pops the
        session from manager._trees.
        """
        d = self._sid_dir(sid)
        try:
            from examples.coding_agent_rl.trajectory_manager_debug import dump_tree_json, dump_tree_txt
        except ImportError:
            logger.warning(
                "on_drain_start(sid=%s): trajectory_manager_debug not importable; "
                "skipping trajectory_tree.{txt,json}",
                sid,
            )
            return
        try:
            (d / "trajectory_tree.txt").write_text(dump_tree_txt(manager, sid), encoding="utf-8")
            _dump_json(
                d / "trajectory_tree.json",
                dump_tree_json(manager, sid, include_messages=True),
            )
        except Exception:
            logger.exception("dump_tree(%s) failed", sid)

    def on_drain_done(self, sid: str, samples) -> None:
        """Called AFTER manager.get_trajectory(sid) returns the samples.

        Dumps decoded samples to trajectory.json (debug-only payload with
        prompt_text / response_text).
        """
        d = self._sid_dir(sid)
        try:
            rendered = [sample_dict_with_text(s, tokenizer=self.tok) for s in samples]
            _dump_json(d / "trajectory.json", rendered)
        except Exception:
            logger.exception("trajectory.json dump for sid=%s failed", sid)


# ---------------- module-level helpers (moved from middleware.py) ---------------- #


def _dump_json(path: Path, obj) -> None:
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _decoded_text_from_sse(sse_bytes: bytes) -> str:
    """Reconstruct the assistant's visible output from a captured SSE stream.

    Human-readable tail only — the canonical wire data is the raw SSE frames
    above this rendering in turn_NNNN_response.sse.
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
    """Render a slime Sample as a JSON-serializable dict with prompt_text /
    response_text decoded via tokenizer.

    Used only by the debug dumper for trajectory.json on disk. The HTTP
    /get_trajectory response intentionally does NOT include these two
    fields (consumers can decode sample.tokens themselves).
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


__all__ = [
    "install",
    "Debug",
    "sample_dict_with_text",
]
