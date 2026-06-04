"""Standalone middleware process around AnthropicAdapter.

This module is OPTIONAL. It wraps :class:`~slime.agent.adapters.AnthropicAdapter`
in a standalone aiohttp server so multiple E2B sandboxes (each with their own
Bearer-token sid) can share one adapter + one TrajectoryManager. Used by
``launch_swe.py`` to fan out N SWE instances against a single host port.

Adds, on top of the adapter:

- ``/get_trajectory``: drain a sid via ``adapter.finish_session(sid)`` and
  return the resulting Sample list as JSON.
- Per-request Claude Code wire scrubbing (billing-header sidechannel +
  mid-list ``role: system`` fold). Mirrors the v1 src_v2 behaviour.
- Per-task SSE capture (debug only): when ``--run-dir`` is set, write the
  raw SSE stream per turn to disk so wire artefacts can be inspected.
- Per-sid turn cap (HTTP 429) so a runaway agent can't burn the budget.

Unlike v1 (src_v2) this middleware no longer proxies sglang /generate --
``AnthropicAdapter`` calls sglang directly and feeds the manager itself. So
the middleware here is just a thin scrubbing + drain layer.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging
import re
from pathlib import Path
from typing import Any

from aiohttp import web
from transformers import AutoTokenizer

from slime.agent.adapters.anthropic import AnthropicAdapter
from slime.utils.types import Sample

logger = logging.getLogger("middleware")


# ---------------------------------------------------------------------------
# SSE capture: per-task isolation via a global wrapper installed once at
# import time. See ``docs/superpowers/specs/2026-06-03-sse-capture-race-fix-design.md``
# for the race that motivated this layout in src_v2 (claude-code 2.1.161 fires
# generateSessionTitle + main agent within the same event loop tick).
#
# Untracked tasks pass through with zero overhead.
# ---------------------------------------------------------------------------

_SSE_CAPTURE: dict[asyncio.Task, list[bytes]] = {}
_orig_stream_write = web.StreamResponse.write


async def _capturing_stream_write(self, data):  # type: ignore[no-untyped-def]
    cap = _SSE_CAPTURE.get(asyncio.current_task())
    if cap is not None:
        cap.append(bytes(data))
    return await _orig_stream_write(self, data)


web.StreamResponse.write = _capturing_stream_write  # install once, never restored


# ---------------------------------------------------------------------------
# Claude-code wire scrubbing
# ---------------------------------------------------------------------------

# Claude Code CLI leaks ``x-anthropic-billing-header: ...cch=<hash>;`` as a text
# block at the top of the system prompt. The cch hash changes per request, so
# without stripping it the rendered system tokens differ every turn and the
# manager tree can't chain consecutive turns together.
_CLAUDE_CODE_BILLING_HEADER_RE = re.compile(
    r"^\s*x-anthropic-billing-header:[^\n]*\n?",
    re.IGNORECASE,
)


def _scrub_billing_header_in_body(body_obj: dict) -> bool:
    """Strip Claude Code's billing-header sidechannel from ``body['system']``.

    Handles both Anthropic shapes (``system: str`` and
    ``system: list[{type:"text",text:"..."}]``).  Mutates ``body_obj`` in
    place; returns True if anything changed.
    """
    sysm = body_obj.get("system")
    changed = False
    if isinstance(sysm, str):
        cleaned = _CLAUDE_CODE_BILLING_HEADER_RE.sub("", sysm)
        if cleaned != sysm:
            body_obj["system"] = cleaned if cleaned.strip() else ""
            changed = True
    elif isinstance(sysm, list):
        new_blocks: list = []
        for block in sysm:
            if not isinstance(block, dict) or block.get("type") != "text":
                new_blocks.append(block)
                continue
            txt = block.get("text") or ""
            cleaned = _CLAUDE_CODE_BILLING_HEADER_RE.sub("", txt)
            if not cleaned.strip():
                # Whole block was the sidechannel — drop it.
                changed = True
                continue
            if cleaned != txt:
                new_block = dict(block)
                new_block["text"] = cleaned
                new_blocks.append(new_block)
                changed = True
            else:
                new_blocks.append(block)
        if changed:
            body_obj["system"] = new_blocks
    return changed


_MID_SYSTEM_WRAP_PREFIX = "<system-reminder>\n"
_MID_SYSTEM_WRAP_SUFFIX = "\n</system-reminder>\n"


def _flatten_anth_text(content: Any) -> str:
    """Best-effort flatten of an Anthropic content value to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for b in content:
        if isinstance(b, dict) and b.get("type") == "text":
            parts.append(b.get("text", ""))
        elif isinstance(b, str):
            parts.append(b)
    return "\n".join(p for p in parts if p)


def _fold_mid_list_system_into_user(body_obj: dict) -> bool:
    """Fold non-leading ``role: system`` messages into a neighbouring user
    message as a ``<system-reminder>`` text block. Mutates ``body_obj`` in
    place.

    claude-code CLI >= 2.1.161 inserts ``{"role":"system","content":"<skills
    list>"}`` in the middle of ``messages``. Qwen3-style chat templates
    reject any system message past index 0 with ``System message must be at
    the beginning.`` This wrap mirrors what claude-code <= 2.1.143 used to do.
    """
    msgs = body_obj.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return False

    system_idx = [i for i, m in enumerate(msgs) if isinstance(m, dict) and m.get("role") == "system" and i > 0]
    if not system_idx:
        return False

    def _promote_to_list(msg: dict) -> list:
        c = msg.get("content")
        if isinstance(c, list):
            return c
        msg["content"] = [{"type": "text", "text": c if isinstance(c, str) else ""}]
        return msg["content"]

    def _wrap(text: str) -> dict:
        return {
            "type": "text",
            "text": _MID_SYSTEM_WRAP_PREFIX + text + _MID_SYSTEM_WRAP_SUFFIX,
        }

    changed = False
    TOMBSTONE: dict = {"__folded__": True}
    for i in system_idx:
        sys_msg = msgs[i]
        wrapped = _wrap(_flatten_anth_text(sys_msg.get("content")))
        target = None
        for j in range(i - 1, -1, -1):
            cand = msgs[j]
            if isinstance(cand, dict) and cand.get("role") == "user":
                target = cand
                _promote_to_list(target).append(wrapped)
                break
        if target is None:
            for j in range(i + 1, len(msgs)):
                cand = msgs[j]
                if isinstance(cand, dict) and cand.get("role") == "user":
                    target = cand
                    _promote_to_list(target).insert(0, wrapped)
                    break
        if target is None:
            msgs[i] = {"role": "user", "content": [wrapped]}
            changed = True
            continue
        msgs[i] = TOMBSTONE
        changed = True

    if changed:
        body_obj["messages"] = [m for m in msgs if m is not TOMBSTONE]
    return changed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--model", required=True, help="HF tokenizer path or repo id")
    p.add_argument(
        "--sglang-url",
        required=True,
        help="Real upstream sglang base URL (the adapter dials it directly).",
    )
    p.add_argument("--tool-parser", default="qwen3_coder")
    p.add_argument("--reasoning-parser", default="qwen3")
    p.add_argument(
        "--run-dir",
        default=None,
        help="If set, import middleware_debug and dump per-turn artefacts "
        "there. Without this flag the middleware writes nothing to disk.",
    )
    p.add_argument(
        "--max-turns-per-sid",
        type=int,
        default=100,
        help="Hard cap on /v1/messages turns per sid. Once a sid hits this "
        "cap, further requests get HTTP 429 so the CC client in the sandbox "
        "exits cleanly. Default 100 stops runaway agents.",
    )
    p.add_argument(
        "--tito-snapshot-min-loss-tokens",
        type=int,
        default=1000,
        help="If a TITO drift would drop >= this many loss_mask=1 tokens, "
        "emit an extra 'snapshot' Sample carrying those tokens before the "
        "main-leaf sample (sibling group_id, equal reward, complementary "
        "loss_mask). Set to 0 to disable. Default 1000.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------


def build_app(args: argparse.Namespace) -> web.Application:
    logger.info("loading tokenizer from %s", args.model)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    adapter = AnthropicAdapter(
        tokenizer=tok,
        sglang_url=args.sglang_url.rstrip("/"),
        tool_parser=args.tool_parser,
        reasoning_parser=args.reasoning_parser,
        tito_snapshot_min_loss_tokens=args.tito_snapshot_min_loss_tokens,
    )

    # Conditional debug install. When --run-dir is set, middleware_debug
    # takes over all per-turn / per-session disk dumps via the hook methods
    # called below. Core middleware never writes to disk on its own.
    DEBUG = None
    if args.run_dir:
        try:
            from examples.coding_agent_rl import middleware_debug  # type: ignore

            DEBUG = middleware_debug.install(args.run_dir, tok, adapter.manager, args)
        except Exception:
            logger.exception(
                "--run-dir=%s set but middleware_debug install failed; " "running without debug dumps",
                args.run_dir,
            )

    # Per-sid turn counters so multiple sandboxes sharing :18080 don't
    # interleave debug-file names.
    turn_counters: dict[str, itertools.count] = {}
    counter_lock = asyncio.Lock()
    opened_sids: set[str] = set()
    sid_first_user: dict[str, str] = {}
    sid_turn_count: dict[str, int] = {}
    max_turns_per_sid: int = args.max_turns_per_sid

    async def _next_turn(sid: str) -> int:
        async with counter_lock:
            ctr = turn_counters.get(sid)
            if ctr is None:
                ctr = itertools.count(1)
                turn_counters[sid] = ctr
            return next(ctr)

    def _sid_of(request: web.Request) -> str:
        auth = request.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            sid = auth[7:].strip()
            if sid:
                return sid
        api_key = request.headers.get("X-Api-Key", "")
        if api_key:
            return api_key.strip()
        return "default"

    def _first_user_prompt(body_obj: dict) -> str:
        for m in body_obj.get("messages") or []:
            if not isinstance(m, dict) or m.get("role") != "user":
                continue
            content = m.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        return block.get("text", "")
        sysm = body_obj.get("system")
        if isinstance(sysm, str):
            return sysm
        return ""

    @web.middleware
    async def dump_mw(request: web.Request, handler):
        if request.path != "/v1/messages":
            return await handler(request)

        sid = _sid_of(request)
        if sid not in opened_sids:
            try:
                adapter.open_session(sid)
            except ValueError:
                pass
            opened_sids.add(sid)

        # Enforce per-sid turn cap. After max_turns_per_sid we return 429 so
        # the cc client in the sandbox exits cleanly.
        prior = sid_turn_count.get(sid, 0)
        if prior >= max_turns_per_sid:
            logger.warning(
                "sid=%s exceeded max_turns_per_sid=%d; returning 429",
                sid,
                max_turns_per_sid,
            )
            return web.json_response(
                {
                    "error": {
                        "type": "rate_limit_error",
                        "message": (
                            f"middleware: sid {sid!r} exceeded " f"max_turns_per_sid={max_turns_per_sid}; killing run"
                        ),
                    }
                },
                status=429,
            )
        sid_turn_count[sid] = prior + 1

        body_bytes = await request.read()
        n = await _next_turn(sid)
        body_obj: dict | None
        try:
            body_obj = json.loads(body_bytes)
        except json.JSONDecodeError:
            body_obj = None
        if DEBUG:
            DEBUG.on_request(sid, n, body_obj, body_bytes)

        # Strip Claude Code's per-request billing-header sidechannel BEFORE
        # the adapter renders prompt_ids. Also fold mid-list ``role: system``
        # messages into a neighbouring user message.
        scrubbed = False
        if body_obj is not None:
            scrubbed |= _scrub_billing_header_in_body(body_obj)
            scrubbed |= _fold_mid_list_system_into_user(body_obj)
        if scrubbed:
            scrubbed_bytes = json.dumps(body_obj, ensure_ascii=False).encode("utf-8")
            request._read_bytes = scrubbed_bytes  # type: ignore[attr-defined]
            if DEBUG:
                DEBUG.on_request_scrubbed(sid, n, body_obj)

        if body_obj is not None and sid not in sid_first_user:
            sid_first_user[sid] = _first_user_prompt(body_obj)

        captured: list[bytes] = []
        task = asyncio.current_task()
        if task is not None:
            _SSE_CAPTURE[task] = captured
        try:
            resp = await handler(request)
        finally:
            if task is not None:
                _SSE_CAPTURE.pop(task, None)

        if DEBUG:
            sse_bytes = b"".join(captured) if captured else b""
            DEBUG.on_sse(sid, n, sse_bytes, resp)
        logger.info("/v1/messages handled sid=%s turn=%d", sid, n)
        return resp

    adapter.app.middlewares.append(dump_mw)

    # ------------------------ /get_trajectory ----------------------------

    async def get_trajectory_route(request: web.Request) -> web.Response:
        """Drain the manager for a sid and return slime Sample list."""
        params = dict(request.query)
        try:
            payload = await request.json() if request.can_read_body else {}
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            params = {**payload, **params}

        sid = str(params.get("sid", "default"))
        try:
            index = int(params.get("index", 0))
        except (TypeError, ValueError):
            index = 0
        try:
            reward = float(params.get("reward", 0.0))
        except (TypeError, ValueError):
            reward = 0.0
        prompt = params.get("prompt") or sid_first_user.get(sid, "")
        label = params.get("label")

        base_sample = Sample(
            index=index,
            group_id=index,
            prompt=prompt,
            label=label,
            metadata={"sid": sid},
        )

        # Best-effort: dump the live tree BEFORE finishing the session, since
        # finish_session pops it (manager.get_trajectory(drop=True)).
        if DEBUG:
            DEBUG.on_drain_start(sid, adapter.manager)

        try:
            samples = await adapter.finish_session(
                sid,
                base_sample=base_sample,
                reward=reward,
            )
        except Exception as e:
            logger.exception("finish_session(%s) failed", sid)
            return web.json_response({"ok": False, "error": str(e)}, status=500)

        if DEBUG:
            DEBUG.on_drain_done(sid, samples)

        rendered = [s.to_dict() if hasattr(s, "to_dict") else dict(s.__dict__) for s in samples]
        return web.json_response(
            {
                "ok": True,
                "sid": sid,
                "num_samples": len(samples),
                "samples": rendered,
            }
        )

    adapter.app.router.add_get("/get_trajectory", get_trajectory_route)
    adapter.app.router.add_post("/get_trajectory", get_trajectory_route)
    # Backwards-compat alias.
    adapter.app.router.add_post("/finish", get_trajectory_route)

    return adapter.app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[middleware %(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    if args.run_dir:
        Path(args.run_dir).mkdir(parents=True, exist_ok=True)
    app = build_app(args)
    logger.info("listening on 0.0.0.0:%d", args.port)
    web.run_app(app, host="0.0.0.0", port=args.port, print=None)


if __name__ == "__main__":
    main()
