import asyncio
import json
import sys
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.agent.adapters import anthropic
from slime.agent.adapters.common import TurnRecord


NUM_GPUS = 0


class ToyTokenizer:
    def __init__(self, outputs: dict[tuple[int, ...], str] | None = None) -> None:
        self.outputs = outputs or {}
        self.rendered: list[tuple[list[dict], list[dict] | None]] = []

    def apply_chat_template(self, messages, tools=None, tokenize=True, add_generation_prompt=True):
        self.rendered.append((list(messages), tools))
        return list(range(1, len(messages) + 2))

    def decode(self, ids, skip_special_tokens=False):
        return self.outputs.get(tuple(ids), "")


class ScriptedTokenizer(ToyTokenizer):
    def __init__(self, prompts: list[list[int]], outputs: dict[tuple[int, ...], str]) -> None:
        super().__init__(outputs)
        self.prompts = [list(prompt) for prompt in prompts]

    def apply_chat_template(self, messages, tools=None, tokenize=True, add_generation_prompt=True):
        self.rendered.append((list(messages), tools))
        assert self.prompts, "unexpected chat-template render"
        return self.prompts.pop(0)


class FakeSGLang:
    def __init__(self, turns: list[list[tuple[float, int]]]) -> None:
        self.turns = [list(turn) for turn in turns]
        self.requests: list[dict] = []
        self.routing_keys: list[str | None] = []

    async def handle_generate(self, request):
        self.routing_keys.append(request.headers.get("X-SMG-Routing-Key"))
        self.requests.append(await request.json())
        assert self.turns, "unexpected /generate call"
        output_token_logprobs = [[logprob, token_id] for logprob, token_id in self.turns.pop(0)]
        return web.json_response(
            {
                "meta_info": {
                    "output_token_logprobs": output_token_logprobs,
                    "finish_reason": {"type": "stop"},
                }
            }
        )


class FakeRequest:
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers


def _parse_sse(raw: str) -> list[tuple[str, object]]:
    events: list[tuple[str, object]] = []
    event_name = "message"
    data_lines: list[str] = []

    def flush() -> None:
        nonlocal event_name, data_lines
        if not data_lines:
            event_name = "message"
            return
        data = "\n".join(data_lines)
        payload: object
        if data == "[DONE]":
            payload = data
        else:
            payload = json.loads(data)
        events.append((event_name, payload))
        event_name = "message"
        data_lines = []

    for line in raw.splitlines():
        if not line:
            flush()
        elif line.startswith("event:"):
            event_name = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())
    flush()
    return events


@pytest.mark.unit
def test_session_id_comes_from_protocol_fields_not_custom_header():
    assert (
        anthropic._request_session_id(FakeRequest({"X-Slime-Session-Id": "custom", "X-Api-Key": "anthropic-key"}))
        == "anthropic-key"
    )
    assert (
        anthropic._request_session_id(
            FakeRequest({"Authorization": "Bearer bearer-session", "X-Api-Key": "anthropic-key"})
        )
        == "bearer-session"
    )


@pytest.mark.unit
def test_anthropic_translation_keeps_tool_results_and_tool_schema():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "plan"},
                {"type": "text", "text": "ok"},
                {"type": "tool_use", "name": "lookup", "input": {"q": "slime"}},
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "u1", "content": "result"}]},
    ]

    translated = anthropic._translate_anthropic(messages, system="sys")
    tools = anthropic._anthropic_tools_to_chat_tools(
        [{"name": "lookup", "description": "search", "input_schema": {"type": "object"}}]
    )

    assert translated == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "ok",
            "reasoning_content": "plan",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "arguments": {"q": "slime"}},
                }
            ],
        },
        {"role": "tool", "content": "result"},
    ]
    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "search",
                "parameters": {"type": "object"},
            },
        }
    ]


# ---------------------------------------------------------------------------
# _is_cc_title_generation_request — detect cc per-session title-gen requests
# (spec: docs/superpowers/specs/2026-06-04-skip-cc-title-gen-from-trajectory-design.md)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_is_cc_title_generation_request_detects_title_gen_body():
    # cc title-gen request: NO tools + system block contains the marker string.
    translated = [
        {
            "role": "system",
            "content": (
                "You are a Claude agent, built on Anthropic's Claude Agent SDK.\n"
                "Generate a concise, sentence-case title (3-7 words) that captures the user's task."
            ),
        },
        {"role": "user", "content": "Read PROBLEM_STATEMENT.md and fix the bug."},
    ]
    tools_schema = None  # cc sends tools=[] -> _anthropic_tools_to_chat_tools returns None
    assert anthropic._is_cc_title_generation_request(translated, tools_schema) is True


@pytest.mark.unit
def test_is_cc_title_generation_request_rejects_main_conversation():
    # cc main agent request: has tools AND no title-gen marker. Must NOT match.
    translated = [
        {
            "role": "system",
            "content": (
                "You are a Claude agent, built on Anthropic's Claude Agent SDK.\n"
                "You are an interactive agent that helps users with software engineering tasks."
            ),
        },
        {"role": "user", "content": "<system-reminder>...</system-reminder>"},
    ]
    tools_schema = [
        {"type": "function", "function": {"name": "Read", "description": "", "parameters": {}}},
        {"type": "function", "function": {"name": "Edit", "description": "", "parameters": {}}},
    ]
    assert anthropic._is_cc_title_generation_request(translated, tools_schema) is False


@pytest.mark.unit
def test_is_cc_title_generation_request_rejects_subagent():
    # cc sub-agent request: has tools AND no title-gen marker. Must NOT match.
    translated = [
        {
            "role": "system",
            "content": (
                "You are a Claude agent, built on Anthropic's Claude Agent SDK.\n"
                "You are a file search specialist for Claude Code, Anthropic's official CLI for Claude."
            ),
        },
        {"role": "user", "content": "Find files related to authentication."},
    ]
    tools_schema = [
        {"type": "function", "function": {"name": "Grep", "description": "", "parameters": {}}},
    ]
    assert anthropic._is_cc_title_generation_request(translated, tools_schema) is False


@pytest.mark.unit
def test_is_cc_title_generation_request_handles_block_list_system_content():
    # cc may send system as a list of blocks (anthropic SDK shape).
    # The marker may live inside one of the blocks; helper must scan all of them.
    translated = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.143"},
                {"type": "text", "text": "You are a Claude agent."},
                {
                    "type": "text",
                    "text": "Generate a concise, sentence-case title (3-7 words) ...",
                },
            ],
        },
        {"role": "user", "content": "hi"},
    ]
    assert anthropic._is_cc_title_generation_request(translated, None) is True


@pytest.mark.unit
def test_is_cc_title_generation_request_rejects_when_tools_present_even_if_marker():
    # AND-conjunction: tools present -> never title-gen, even if marker text leaks in.
    translated = [
        {
            "role": "system",
            "content": "Generate a concise, sentence-case title (3-7 words) ...",
        },
        {"role": "user", "content": "hi"},
    ]
    tools_schema = [{"type": "function", "function": {"name": "Read", "description": "", "parameters": {}}}]
    assert anthropic._is_cc_title_generation_request(translated, tools_schema) is False


# ---------------------------------------------------------------------------
# _handle_request: title-gen requests skip append_turn but still fire hook
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_anthropic_handle_request_skips_append_turn_for_title_gen(monkeypatch):
    """A cc title-gen request must NOT enter manager._trees, but the
    on_turn_appended hook MUST still fire so per-turn dumps (openai.json,
    etc.) keep being written."""

    async def fake_generate(prompt_ids, session, body, app, **kwargs):
        return TurnRecord(
            prompt_ids=list(prompt_ids),
            output_ids=[42, 43, 44],
            finish_reason="stop",
            output_log_probs=[-0.1, -0.2, -0.3],
        )

    monkeypatch.setattr(anthropic, "call_sglang_generate", fake_generate)

    hook_calls: list[tuple] = []

    def hook(sid, prompt_messages, tools, response_message, prompt_ids, response_ids, finish_reason):
        hook_calls.append((sid, len(prompt_ids), len(response_ids), finish_reason))

    tok = ToyTokenizer(outputs={(42, 43, 44): '{"title": "Fix the bug"}<|im_end|>'})

    adapter = anthropic.AnthropicAdapter(
        tokenizer=tok,
        sglang_url="http://127.0.0.1:1",
        on_turn_appended=hook,
    )

    # Build a cc-style title-gen request body.
    title_gen_body = {
        "messages": [{"role": "user", "content": "Read PROBLEM_STATEMENT.md and fix."}],
        "system": [
            {"type": "text", "text": "You are a Claude agent."},
            {
                "type": "text",
                "text": "Generate a concise, sentence-case title (3-7 words) ...",
            },
        ],
        "tools": [],
        "max_tokens": 512,
        "stream": False,
    }

    async def run():
        app = adapter.app
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            r = await client.post(
                "/v1/messages",
                headers={"Authorization": "Bearer test-title-gen-sid"},
                json=title_gen_body,
            )
            assert r.status == 200, await r.text()
        finally:
            await client.close()

    asyncio.run(run())

    # The manager MUST NOT have learned about this sid (no tree was opened).
    assert (
        "test-title-gen-sid" not in adapter.manager._trees
    ), f"title-gen request leaked into manager: trees={list(adapter.manager._trees)}"

    # The hook MUST have fired exactly once with this sid.
    assert len(hook_calls) == 1, f"expected 1 hook call, got {hook_calls}"
    assert hook_calls[0][0] == "test-title-gen-sid"
    assert hook_calls[0][2] == 3  # response_ids length matches fake_generate output


@pytest.mark.unit
def test_anthropic_handle_request_still_appends_main_conversation(monkeypatch):
    """Sanity counter-test: a normal (non-title-gen) request MUST still
    populate manager._trees and fire the hook. Prevents an over-broad guard
    from silently dropping real conversation turns."""

    async def fake_generate(prompt_ids, session, body, app, **kwargs):
        return TurnRecord(
            prompt_ids=list(prompt_ids),
            output_ids=[7, 8, 9],
            finish_reason="stop",
            output_log_probs=[-0.1, -0.2, -0.3],
        )

    monkeypatch.setattr(anthropic, "call_sglang_generate", fake_generate)

    hook_calls: list[tuple] = []

    def hook(sid, *rest):
        hook_calls.append(sid)

    tok = ToyTokenizer(outputs={(7, 8, 9): "hello"})

    adapter = anthropic.AnthropicAdapter(
        tokenizer=tok,
        sglang_url="http://127.0.0.1:1",
        on_turn_appended=hook,
    )

    main_body = {
        "messages": [{"role": "user", "content": "Run the tests."}],
        "system": [
            {"type": "text", "text": "You are an interactive agent."},
        ],
        "tools": [
            {
                "name": "Read",
                "description": "Read a file",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
        "max_tokens": 8000,
        "stream": False,
    }

    async def run():
        app = adapter.app
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            r = await client.post(
                "/v1/messages",
                headers={"Authorization": "Bearer test-main-sid"},
                json=main_body,
            )
            assert r.status == 200, await r.text()
        finally:
            await client.close()

    asyncio.run(run())

    assert (
        "test-main-sid" in adapter.manager._trees
    ), f"main conversation did NOT reach manager: trees={list(adapter.manager._trees)}"
    assert hook_calls == ["test-main-sid"], hook_calls


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
