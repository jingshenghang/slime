"""TDD tests for AnthropicAdapter wire-scrubbing behaviours that used to live
in examples/coding_agent_rl/middleware.py.

These behaviours must apply inside the in-process adapter so the 4-node RL
training path (which talks to AnthropicAdapter directly, not through the
middleware subprocess) sees identical normalised input.

Naming convention follows tests/test_agent_adapters.py: synchronous test
functions run ``asyncio.run(run_case())`` internally; fake sglang upstream is
mounted on an in-process TestServer; the adapter app is wrapped with
TestClient. We reuse ScriptedTokenizer / FakeSGLang directly via
sibling-module import.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


# Sibling-module import of fixtures.
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))
from test_agent_adapters import FakeSGLang, ScriptedTokenizer  # noqa: E402


# ---------------------------------------------------------------------------
# Behaviour 1: billing-header scrub
# ---------------------------------------------------------------------------


def test_scrub_billing_header_from_system_string():
    """A string ``system`` whose first line is the Claude Code billing header
    must have that line stripped; the remaining text is preserved."""
    from slime.agent.adapters.anthropic import _scrub_claude_code_billing_header_in_body

    body = {
        "system": "x-anthropic-billing-header: cch=deadbeef;\nYou are a helpful assistant.",
    }
    changed = _scrub_claude_code_billing_header_in_body(body)
    assert changed is True
    assert body["system"] == "You are a helpful assistant."


def test_scrub_billing_header_from_system_block_list_drops_pure_header_block():
    """A block-list ``system`` containing one block that is *only* the billing
    header must drop that block entirely; other text blocks survive."""
    from slime.agent.adapters.anthropic import _scrub_claude_code_billing_header_in_body

    body = {
        "system": [
            {"type": "text", "text": "x-anthropic-billing-header: cch=abc123;\n"},
            {"type": "text", "text": "You are a helpful assistant."},
        ],
    }
    changed = _scrub_claude_code_billing_header_in_body(body)
    assert changed is True
    assert body["system"] == [{"type": "text", "text": "You are a helpful assistant."}]


def test_scrub_billing_header_from_system_block_keeps_text_when_mixed():
    """A block that has the billing header followed by other text must keep
    the other text and only strip the header line."""
    from slime.agent.adapters.anthropic import _scrub_claude_code_billing_header_in_body

    body = {
        "system": [
            {
                "type": "text",
                "text": "x-anthropic-billing-header: cch=xyz;\nReal system prompt here.",
            },
        ],
    }
    changed = _scrub_claude_code_billing_header_in_body(body)
    assert changed is True
    assert body["system"] == [{"type": "text", "text": "Real system prompt here."}]


def test_scrub_billing_header_no_op_when_absent():
    """When no billing header is present the body must be untouched and the
    helper must report ``False``."""
    from slime.agent.adapters.anthropic import _scrub_claude_code_billing_header_in_body

    body = {"system": "Plain system prompt."}
    changed = _scrub_claude_code_billing_header_in_body(body)
    assert changed is False
    assert body["system"] == "Plain system prompt."


# ---------------------------------------------------------------------------
# Behaviour 2: mid-list ``role: system`` fold-into-user
# ---------------------------------------------------------------------------


def test_fold_mid_list_system_appends_to_preceding_user():
    """Qwen3 chat templates refuse system messages past index 0. A mid-list
    ``role: system`` must be folded into the *preceding* user message as a
    ``<system-reminder>`` text block, and removed from ``messages``."""
    from slime.agent.adapters.anthropic import _fold_mid_list_system_into_user

    body = {
        "messages": [
            {"role": "system", "content": "leading system stays"},
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "extra skill reminder"},
            {"role": "assistant", "content": "hi"},
        ]
    }
    changed = _fold_mid_list_system_into_user(body)
    assert changed is True
    msgs = body["messages"]
    assert len(msgs) == 3
    assert msgs[0] == {"role": "system", "content": "leading system stays"}
    user_msg = msgs[1]
    assert user_msg["role"] == "user"
    assert isinstance(user_msg["content"], list)
    assert user_msg["content"][0] == {"type": "text", "text": "hello"}
    folded_block = user_msg["content"][1]
    assert folded_block["type"] == "text"
    assert "<system-reminder>" in folded_block["text"]
    assert "extra skill reminder" in folded_block["text"]
    assert "</system-reminder>" in folded_block["text"]


def test_fold_mid_list_system_falls_back_to_following_user_when_no_prior_user():
    """If there's no user message *before* a mid-list system, fall back to the
    next user message and insert the wrapped block at its head."""
    from slime.agent.adapters.anthropic import _fold_mid_list_system_into_user

    body = {
        "messages": [
            {"role": "system", "content": "leading"},
            {"role": "system", "content": "stray skill"},
            {"role": "user", "content": "first user msg"},
        ]
    }
    changed = _fold_mid_list_system_into_user(body)
    assert changed is True
    assert len(body["messages"]) == 2
    assert body["messages"][0] == {"role": "system", "content": "leading"}
    fallback_user = body["messages"][1]
    assert fallback_user["role"] == "user"
    first_block = fallback_user["content"][0]
    assert "<system-reminder>" in first_block["text"]
    assert "stray skill" in first_block["text"]


def test_fold_mid_list_system_no_op_when_no_mid_system():
    """When all system messages are at index 0 the body must be untouched."""
    from slime.agent.adapters.anthropic import _fold_mid_list_system_into_user

    body = {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]
    }
    original = [dict(m) for m in body["messages"]]
    changed = _fold_mid_list_system_into_user(body)
    assert changed is False
    assert body["messages"] == original


# ---------------------------------------------------------------------------
# Behaviour 3: per-sid turn cap (HTTP 429) exposed through the adapter
# ---------------------------------------------------------------------------


def _build_adapter_with_upstream(max_turns_per_sid):
    """Helper: spins up an in-process fake sglang and an AnthropicAdapter
    that points at it. Returns (adapter, fake_sglang, upstream_server)."""
    from slime.agent.adapters import anthropic as anth

    upstream = FakeSGLang(
        [
            [(-0.1, 200), (-0.2, 201)],
            [(-0.1, 200), (-0.2, 201)],
            [(-0.1, 200), (-0.2, 201)],
            [(-0.1, 200), (-0.2, 201)],
            [(-0.1, 200), (-0.2, 201)],
        ]
    )
    return upstream, anth


def test_per_sid_turn_cap_returns_429_after_threshold():
    """``AnthropicAdapter(max_turns_per_sid=2)`` lets two /v1/messages calls
    through for a single sid; the third returns HTTP 429 with the
    rate_limit_error shape."""
    from slime.agent.adapters import anthropic as anth

    async def run_case():
        upstream = FakeSGLang(
            [
                [(-0.1, 200), (-0.2, 201)],
                [(-0.1, 200), (-0.2, 201)],
                [(-0.1, 200), (-0.2, 201)],
            ]
        )
        upstream_app = web.Application()
        upstream_app.router.add_post("/generate", upstream.handle_generate)
        upstream_server = TestServer(upstream_app)
        await upstream_server.start_server()

        tok = ScriptedTokenizer(
            prompts=[[1, 2], [1, 2, 200, 201, 3], [1, 2, 200, 201, 3, 4]],
            outputs={(200, 201): "hi"},
        )
        adapter = anth.AnthropicAdapter(
            tokenizer=tok,
            sglang_url=str(upstream_server.make_url("")).rstrip("/"),
            max_turns_per_sid=2,
        )
        client = TestClient(TestServer(adapter.app))
        await client.start_server()
        try:
            headers = {"Authorization": "Bearer sid-cap-test"}
            payload = {
                "model": "actor",
                "max_tokens": 5,
                "messages": [{"role": "user", "content": "hi"}],
            }
            r1 = await client.post("/v1/messages", json=payload, headers=headers)
            assert r1.status == 200, await r1.text()

            payload2 = {
                "model": "actor",
                "max_tokens": 5,
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hi"},
                    {"role": "user", "content": "again"},
                ],
            }
            r2 = await client.post("/v1/messages", json=payload2, headers=headers)
            assert r2.status == 200, await r2.text()

            r3 = await client.post("/v1/messages", json=payload2, headers=headers)
            assert r3.status == 429, await r3.text()
            body = await r3.json()
            assert body["error"]["type"] == "rate_limit_error"
            assert "max_turns_per_sid=2" in body["error"]["message"]
        finally:
            await client.close()
            await upstream_server.close()

    asyncio.run(run_case())


def test_per_sid_turn_cap_disabled_when_none():
    """``AnthropicAdapter(max_turns_per_sid=None)`` must accept the kwarg and
    impose NO cap — 5 sequential calls all return 200.

    Pairs with ``test_per_sid_turn_cap_returns_429_after_threshold``: that test
    proves the cap works at threshold N; this test proves None disables it.
    Both must reference the new ctor kwarg, otherwise ``None`` could pass
    trivially before the feature exists.
    """
    from slime.agent.adapters import anthropic as anth

    async def run_case():
        upstream = FakeSGLang([[(-0.1, 200), (-0.2, 201)]] * 5)
        upstream_app = web.Application()
        upstream_app.router.add_post("/generate", upstream.handle_generate)
        upstream_server = TestServer(upstream_app)
        await upstream_server.start_server()

        tok = ScriptedTokenizer(
            prompts=[[i + 1, i + 2] for i in range(5)],
            outputs={(200, 201): "ok"},
        )
        # Explicitly pass max_turns_per_sid=None so the ctor must accept it.
        adapter = anth.AnthropicAdapter(
            tokenizer=tok,
            sglang_url=str(upstream_server.make_url("")).rstrip("/"),
            max_turns_per_sid=None,
        )
        client = TestClient(TestServer(adapter.app))
        await client.start_server()
        try:
            headers = {"Authorization": "Bearer sid-nocap"}
            for i in range(5):
                payload = {
                    "model": "actor",
                    "max_tokens": 5,
                    "messages": [{"role": "user", "content": f"hi-{i}"}],
                }
                r = await client.post("/v1/messages", json=payload, headers=headers)
                assert r.status == 200, await r.text()
        finally:
            await client.close()
            await upstream_server.close()

    asyncio.run(run_case())
