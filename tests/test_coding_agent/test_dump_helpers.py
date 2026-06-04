"""Pragmatic-TDD tests for the dump middleware + hook surface in
``tests/test_coding_agent/_dump_helpers.py``.

Each behavior is added one RED-then-GREEN cycle at a time. The hooks
correspond to the per-turn / per-session disk artefacts that
``examples/coding_agent_rl/middleware_debug.py`` historically produced; we
move them under ``tests/`` because they are only exercised by the SWE e2e
test (Phase 2 of the trajectory_manager migration plan).

Conventions copied from ``tests/test_agent_adapters.py``:
  * synchronous ``test_*`` functions run ``asyncio.run(run_case())``
    internally (the repo does not have ``pytest-asyncio``).
  * the in-process adapter app is wrapped with
    ``aiohttp.test_utils.TestServer`` / ``TestClient``; a ``FakeSGLang``
    upstream is mounted on a second ``TestServer``.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

# Sibling-module import of fixtures from tests/test_agent_adapters.py.
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))
from test_agent_adapters import FakeSGLang, ScriptedTokenizer  # noqa: E402

# ---------------------------------------------------------------------------
# Behavior 1: install_dump_layer factory entry
# ---------------------------------------------------------------------------


def test_install_dump_layer_returns_debug_and_creates_run_dir(tmp_path):
    """``install_dump_layer(adapter, run_dir=..., tokenizer=...)`` must:

    * accept an ``AnthropicAdapter`` instance and create ``run_dir`` if it
      doesn't exist;
    * return an object exposing the 7 hook methods;
    * register an aiohttp middleware on ``adapter.app`` (so the dump
      wrapper actually gets exercised for /v1/messages requests);
    * wire ``adapter.on_turn_appended`` so the adapter-side per-turn dump
      keeps firing into the returned Debug.
    """
    from tests.test_coding_agent._dump_helpers import Debug, install_dump_layer

    from slime.agent.adapters import anthropic as anth

    tok = ScriptedTokenizer(prompts=[], outputs={})
    adapter = anth.AnthropicAdapter(tokenizer=tok, sglang_url="http://unused")

    run_dir = tmp_path / "run"
    assert not run_dir.exists()

    n_mw_before = len(adapter.app.middlewares)

    handle = install_dump_layer(adapter, run_dir=run_dir, tokenizer=tok)

    assert isinstance(handle, Debug)
    assert run_dir.is_dir()
    # Hook methods exist with the documented names.
    for hook in (
        "on_request",
        "on_request_scrubbed",
        "on_sse",
        "on_sglang_pair",
        "on_turn_appended",
        "on_drain_start",
        "on_drain_done",
    ):
        assert callable(getattr(handle, hook)), f"missing hook: {hook}"

    # A dump middleware MUST be appended to the adapter's app.
    assert len(adapter.app.middlewares) == n_mw_before + 1

    # Adapter-side per-turn hook MUST be wired so append_turn payloads land
    # in turn_NNNN_openai.json. The exact attribute name on Debug is an
    # implementation detail; what matters is that the adapter's hook is
    # set to something callable that routes into the handle.
    assert callable(adapter.on_turn_appended)


# ---------------------------------------------------------------------------
# Behavior 2: on_request — JSON body + raw fallback
# ---------------------------------------------------------------------------


def test_on_request_writes_json_when_body_obj_present(tmp_path):
    """When ``body_obj`` is a dict, ``on_request`` writes
    ``turn_NNNN_request.json`` containing that body and does NOT write the
    ``.raw`` companion."""
    from tests.test_coding_agent._dump_helpers import Debug

    tok = ScriptedTokenizer(prompts=[], outputs={})
    handle = Debug(tmp_path, tokenizer=tok)

    body_obj = {"model": "actor", "messages": [{"role": "user", "content": "hi"}]}
    handle.on_request("sid-A", 7, body_obj, b"original bytes ignored when obj set")

    sid_dir = tmp_path / "sid-A"
    assert (sid_dir / "turn_0007_request.json").is_file()
    assert not (sid_dir / "turn_0007_request.raw").exists()
    loaded = json.loads((sid_dir / "turn_0007_request.json").read_text())
    assert loaded == body_obj


def test_on_request_writes_raw_when_body_obj_none(tmp_path):
    """When ``body_obj`` is None (e.g. body wasn't valid JSON), the raw
    bytes go to ``turn_NNNN_request.raw`` and no ``.json`` is produced."""
    from tests.test_coding_agent._dump_helpers import Debug

    tok = ScriptedTokenizer(prompts=[], outputs={})
    handle = Debug(tmp_path, tokenizer=tok)

    raw = b"\x00not-valid-json\xff"
    handle.on_request("sid-B", 1, None, raw)

    sid_dir = tmp_path / "sid-B"
    assert (sid_dir / "turn_0001_request.raw").read_bytes() == raw
    assert not (sid_dir / "turn_0001_request.json").exists()


# ---------------------------------------------------------------------------
# Behavior 3: on_request_scrubbed
# ---------------------------------------------------------------------------


def test_on_request_scrubbed_writes_scrubbed_json(tmp_path):
    """Scrubbed body (after billing-header strip / system-fold) is written
    to ``turn_NNNN_request_scrubbed.json`` so a diff vs the original
    request.json shows exactly what the adapter changed."""
    from tests.test_coding_agent._dump_helpers import Debug

    tok = ScriptedTokenizer(prompts=[], outputs={})
    handle = Debug(tmp_path, tokenizer=tok)

    scrubbed = {"messages": [{"role": "user", "content": "clean"}], "system": "kept"}
    handle.on_request_scrubbed("sid-S", 3, scrubbed)
    out = tmp_path / "sid-S" / "turn_0003_request_scrubbed.json"
    assert out.is_file()
    assert json.loads(out.read_text()) == scrubbed


# ---------------------------------------------------------------------------
# Behavior 4: on_sse — three branches (sse bytes, json body, raw bytes body)
# ---------------------------------------------------------------------------


def test_on_sse_writes_sse_file_with_decoded_tail(tmp_path):
    """When ``sse_bytes`` is non-empty, the raw frames go to
    ``turn_NNNN_response.sse`` and a decoded human-readable tail is
    appended (text + thinking + tool_use reconstructed from delta events)."""
    from tests.test_coding_agent._dump_helpers import Debug

    tok = ScriptedTokenizer(prompts=[], outputs={})
    handle = Debug(tmp_path, tokenizer=tok)

    sse = (
        b"event: content_block_start\n"
        b'data: {"type":"content_block_start","index":0,'
        b'"content_block":{"type":"text","text":""}}\n\n'
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":0,'
        b'"delta":{"type":"text_delta","text":"hello"}}\n\n'
        b"event: content_block_start\n"
        b'data: {"type":"content_block_start","index":1,'
        b'"content_block":{"type":"tool_use","name":"lookup","id":"toolu_1"}}\n\n'
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":1,'
        b'"delta":{"type":"input_json_delta","partial_json":"{\\"q\\": \\"slime\\"}"}}\n\n'
    )
    handle.on_sse("sid-X", 5, sse, resp=None)

    sse_path = tmp_path / "sid-X" / "turn_0005_response.sse"
    assert sse_path.is_file()
    out = sse_path.read_bytes()
    # Raw frames are preserved at the head; decoded tail is appended.
    assert out.startswith(sse)
    tail = out[len(sse) :].decode("utf-8")
    assert "decoded assistant output" in tail
    assert "[text]\nhello" in tail
    assert "[tool_use name=lookup id=toolu_1]" in tail
    assert '{"q": "slime"}' in tail


def test_on_sse_writes_json_for_non_stream_response(tmp_path):
    """When ``sse_bytes`` is empty AND the resp is a JSON web.Response, the
    body is decoded and stored at ``turn_NNNN_response.json``."""
    from tests.test_coding_agent._dump_helpers import Debug

    tok = ScriptedTokenizer(prompts=[], outputs={})
    handle = Debug(tmp_path, tokenizer=tok)

    payload = {"id": "msg_x", "content": [{"type": "text", "text": "hi"}]}
    resp = web.json_response(payload)
    handle.on_sse("sid-J", 2, b"", resp)
    out = tmp_path / "sid-J" / "turn_0002_response.json"
    assert out.is_file()
    assert json.loads(out.read_text()) == payload


def test_on_sse_writes_raw_for_non_json_response_body(tmp_path):
    """A web.Response whose body is bytes-but-not-JSON falls back to
    ``turn_NNNN_response.raw``."""
    from tests.test_coding_agent._dump_helpers import Debug

    tok = ScriptedTokenizer(prompts=[], outputs={})
    handle = Debug(tmp_path, tokenizer=tok)

    resp = web.Response(body=b"\xffnot-json")
    handle.on_sse("sid-R", 4, b"", resp)
    out = tmp_path / "sid-R" / "turn_0004_response.raw"
    assert out.read_bytes() == b"\xffnot-json"
    assert not (tmp_path / "sid-R" / "turn_0004_response.json").exists()


# ---------------------------------------------------------------------------
# Behavior 5: on_sglang_pair
# ---------------------------------------------------------------------------


def test_on_sglang_pair_writes_both_files(tmp_path):
    """Both halves of the sglang wire pair land on disk under
    ``turn_NNNN_sglang_request.json`` and ``turn_NNNN_sglang_response.json``."""
    from tests.test_coding_agent._dump_helpers import Debug

    tok = ScriptedTokenizer(prompts=[], outputs={})
    handle = Debug(tmp_path, tokenizer=tok)

    req = {"input_ids": [1, 2, 3], "sampling_params": {"max_new_tokens": 4}}
    resp = {"meta_info": {"output_token_logprobs": [[-0.1, 99]], "finish_reason": {"type": "stop"}}}
    handle.on_sglang_pair("sid-G", 9, req, resp)

    rp = tmp_path / "sid-G" / "turn_0009_sglang_request.json"
    sp = tmp_path / "sid-G" / "turn_0009_sglang_response.json"
    assert json.loads(rp.read_text()) == req
    assert json.loads(sp.read_text()) == resp


def test_on_sglang_pair_skips_missing_half(tmp_path):
    """When one side is ``None`` and no raw bytes fallback is given, that
    file is simply not written; the other side still lands."""
    from tests.test_coding_agent._dump_helpers import Debug

    tok = ScriptedTokenizer(prompts=[], outputs={})
    handle = Debug(tmp_path, tokenizer=tok)

    handle.on_sglang_pair("sid-G2", 1, None, {"only": "resp"})
    sid_dir = tmp_path / "sid-G2"
    assert not (sid_dir / "turn_0001_sglang_request.json").exists()
    assert json.loads((sid_dir / "turn_0001_sglang_response.json").read_text()) == {"only": "resp"}


# ---------------------------------------------------------------------------
# Behavior 6: on_turn_appended — OpenAI-shape dump
# ---------------------------------------------------------------------------


def test_on_turn_appended_writes_openai_json(tmp_path):
    """``on_turn_appended`` writes ``turn_NNNN_openai.json`` carrying the
    OpenAI-shape view of the manager's append_turn payload (full
    prompt_messages, tools, response_message, plus prompt/response id
    *lengths* for compactness)."""
    from tests.test_coding_agent._dump_helpers import Debug

    tok = ScriptedTokenizer(prompts=[], outputs={})
    handle = Debug(tmp_path, tokenizer=tok)

    prompt_messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "Read"}}]
    response_message = {"role": "assistant", "content": "ok"}
    prompt_ids = [1, 2, 3, 4]
    response_ids = [10, 11]
    handle.on_turn_appended(
        "sid-O",
        12,
        prompt_messages,
        tools,
        response_message,
        prompt_ids,
        response_ids,
        "stop",
    )

    out = tmp_path / "sid-O" / "turn_0012_openai.json"
    payload = json.loads(out.read_text())
    assert payload["sid"] == "sid-O"
    assert payload["turn"] == 12
    assert payload["prompt_messages"] == prompt_messages
    assert payload["tools"] == tools
    assert payload["response_message"] == response_message
    assert payload["prompt_ids_len"] == 4
    assert payload["response_ids_len"] == 2
    assert payload["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# Behavior 7: on_drain_start — tree snapshot BEFORE drain pops the tree
# ---------------------------------------------------------------------------


def test_on_drain_start_writes_tree_txt_and_json(tmp_path):
    """``on_drain_start(sid, manager)`` dumps the live tree to
    ``trajectory_tree.txt`` and ``trajectory_tree.json``. Must be called
    BEFORE ``manager.get_trajectory(sid, drop=True)`` because the drain
    pops the per-sid tree."""
    from tests.test_coding_agent._dump_helpers import Debug

    from slime.agent.trajectory_manager import TrajectoryManager

    mgr = TrajectoryManager()
    sid = "sid-T"
    mgr.append_turn(
        sid,
        prompt_messages=[
            {"role": "system", "content": "S"},
            {"role": "user", "content": "u"},
        ],
        tools=None,
        prompt_ids=[1, 2, 3],
        response_ids=[10, 11],
        response_logprobs=[-0.5, -0.6],
        response_message={"role": "assistant", "content": "a"},
        finish_reason="stop",
    )

    tok = ScriptedTokenizer(prompts=[], outputs={})
    handle = Debug(tmp_path, tokenizer=tok)
    handle.on_drain_start(sid, mgr)

    txt = (tmp_path / sid / "trajectory_tree.txt").read_text()
    assert f"session={sid}" in txt
    assert "turn=1" in txt

    j = json.loads((tmp_path / sid / "trajectory_tree.json").read_text())
    assert j["sid"] == sid
    assert j["found"] is True
    assert j["turns"] == 1

    # include_messages=True must be on so raw messages survive round-trip.
    def _find_asst(node):
        if node.get("role") == "assistant":
            return node
        for c in node.get("children", []):
            r = _find_asst(c)
            if r is not None:
                return r
        return None

    asst = _find_asst(j["root"])
    assert asst is not None and "messages" in asst


# ---------------------------------------------------------------------------
# Behavior 8: on_drain_done — decoded sample dicts in trajectory.json
# ---------------------------------------------------------------------------


def test_on_drain_done_writes_trajectory_json_with_decoded_text(tmp_path):
    """``on_drain_done(sid, samples)`` writes ``trajectory.json``: a list
    of per-sample dicts enriched with ``prompt_text`` / ``response_text``
    decoded via the configured tokenizer. The over-the-wire
    /get_trajectory body does NOT carry these fields — this is debug-only."""
    from tests.test_coding_agent._dump_helpers import Debug

    from slime.utils.types import Sample

    class _Tok:
        def decode(self, ids, skip_special_tokens=False):
            if not ids:
                return ""
            return "<" + ",".join(str(x) for x in ids) + ">"

    handle = Debug(tmp_path, tokenizer=_Tok())

    # tokens = prompt (0..3) + response (4..6); response_length=3.
    sample = Sample(
        index=0,
        rollout_id=0,
        prompt="ignored",
        tokens=[100, 101, 102, 103, 200, 201, 202],
        response_length=3,
    )
    handle.on_drain_done("sid-D", [sample])

    out = tmp_path / "sid-D" / "trajectory.json"
    payload = json.loads(out.read_text())
    assert isinstance(payload, list) and len(payload) == 1
    one = payload[0]
    # tokens[:4] -> prompt, tokens[4:] -> response under response_length=3
    assert one["prompt_text"] == "<100,101,102,103>"
    assert one["response_text"] == "<200,201,202>"
    # Sanity: tokens / response_length must survive the dict round-trip.
    assert one["tokens"] == [100, 101, 102, 103, 200, 201, 202]
    assert one["response_length"] == 3


# ---------------------------------------------------------------------------
# Behavior 9: sid_dump_dir override routes a sid's artefacts to an absolute dir
# ---------------------------------------------------------------------------


def test_sid_dump_dir_override_routes_artifacts(tmp_path):
    """When a sid has an entry in the ``sid_dump_dir`` mapping passed at
    install time, all of its artefacts land directly in that absolute
    directory instead of under ``run_dir/<sid>/``."""
    from tests.test_coding_agent._dump_helpers import Debug

    inst_dir = tmp_path / "inst-A"
    sid_dump_dir = {"sid-override": str(inst_dir)}
    tok = ScriptedTokenizer(prompts=[], outputs={})
    handle = Debug(tmp_path / "run", tokenizer=tok, sid_dump_dir=sid_dump_dir)

    handle.on_request_scrubbed("sid-override", 1, {"x": 1})
    handle.on_request_scrubbed("sid-default", 1, {"y": 2})

    assert (inst_dir / "turn_0001_request_scrubbed.json").is_file()
    assert (tmp_path / "run" / "sid-default" / "turn_0001_request_scrubbed.json").is_file()
    # The override sid MUST NOT have ALSO produced a fallback file.
    assert not (tmp_path / "run" / "sid-override").exists()


# ---------------------------------------------------------------------------
# Behavior 10: end-to-end integration via build_dump_middleware
# ---------------------------------------------------------------------------


def test_dump_middleware_captures_streaming_and_assigns_per_sid_turn_numbers(tmp_path):
    """End-to-end smoke: install the dump layer on a real AnthropicAdapter
    pointed at a fake sglang upstream, hit /v1/messages twice with the
    same sid (streaming=True), then verify:

      * turn numbers are per-sid monotonic (0001, 0002);
      * each turn produces request.json + response.sse + openai.json;
      * the captured SSE includes ``message_start`` / ``message_stop``
        frames produced by the adapter's stream path.
    """
    from tests.test_coding_agent._dump_helpers import install_dump_layer

    from slime.agent.adapters import anthropic as anth

    async def run_case():
        upstream = FakeSGLang(
            [
                [(-0.1, 200), (-0.2, 201)],
                [(-0.1, 200), (-0.2, 201)],
            ]
        )
        upstream_app = web.Application()
        upstream_app.router.add_post("/generate", upstream.handle_generate)
        upstream_server = TestServer(upstream_app)
        await upstream_server.start_server()

        tok = ScriptedTokenizer(
            prompts=[[1, 2], [1, 2, 3, 4]],
            outputs={(200, 201): "hello"},
        )
        adapter = anth.AnthropicAdapter(
            tokenizer=tok,
            sglang_url=str(upstream_server.make_url("")).rstrip("/"),
        )
        run_dir = tmp_path / "run"
        handle = install_dump_layer(adapter, run_dir=run_dir, tokenizer=tok)

        client = TestClient(TestServer(adapter.app))
        await client.start_server()
        try:
            headers = {"Authorization": "Bearer sid-mw"}
            for i in range(2):
                r = await client.post(
                    "/v1/messages",
                    headers=headers,
                    json={
                        "model": "actor",
                        "stream": True,
                        "max_tokens": 5,
                        "messages": [{"role": "user", "content": f"hi-{i}"}],
                    },
                )
                # Drain the SSE so the server-side write coroutine completes.
                _ = await r.text()
                assert r.status == 200
        finally:
            await client.close()
            await upstream_server.close()

        sid_dir = run_dir / "sid-mw"
        # Two turn requests with monotonic per-sid numbering.
        assert (sid_dir / "turn_0001_request.json").is_file()
        assert (sid_dir / "turn_0002_request.json").is_file()
        # SSE captured for both turns.
        sse1 = (sid_dir / "turn_0001_response.sse").read_bytes()
        sse2 = (sid_dir / "turn_0002_response.sse").read_bytes()
        for sse in (sse1, sse2):
            assert b"message_start" in sse
            assert b"message_stop" in sse
            assert b"decoded assistant output" in sse  # decoded tail present
        # openai.json from the adapter-side on_turn_appended bridge must also
        # exist for both turns (proves the install correctly wired the hook).
        assert (sid_dir / "turn_0001_openai.json").is_file()
        assert (sid_dir / "turn_0002_openai.json").is_file()
        # Sanity: handle still resolves the sid directory under run_dir.
        assert handle._sid_dir("sid-mw") == sid_dir

    asyncio.run(run_case())


def test_dump_middleware_skips_non_messages_paths(tmp_path):
    """A request to ``/healthz`` (or any non-/v1/messages path) must not
    create a turn dump or bump the turn counter."""
    from tests.test_coding_agent._dump_helpers import install_dump_layer

    from slime.agent.adapters import anthropic as anth

    async def run_case():
        tok = ScriptedTokenizer(prompts=[], outputs={})
        adapter = anth.AnthropicAdapter(tokenizer=tok, sglang_url="http://unused")
        run_dir = tmp_path / "run"
        install_dump_layer(adapter, run_dir=run_dir, tokenizer=tok)

        client = TestClient(TestServer(adapter.app))
        await client.start_server()
        try:
            r = await client.get(
                "/healthz",
                headers={"Authorization": "Bearer sid-health"},
            )
            assert r.status == 200
        finally:
            await client.close()

        # No per-sid dir should have been created — the middleware short-
        # circuited before touching the disk.
        assert not (run_dir / "sid-health").exists()

    asyncio.run(run_case())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
