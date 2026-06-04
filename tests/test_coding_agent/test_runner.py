"""TDD tests for tests/test_coding_agent/_runner.py.

Phase 4 of the migration. Covers the pure helpers that used to live at the
top of ``launch_swe.py``:

* ``normalize_sample(raw)``: project a slime-SWE jsonl row to the small dict
  that ``run_one_instance`` consumes.
* ``load_dataset(path, offset, limit)``: read N rows from a jsonl, applying
  offset / limit / blank-line / bad-json skipping.
* ``ensure_e2b_env()``: inject a syntactically-valid placeholder E2B_API_KEY
  when the real one is missing/malformed, plus standard sandbox metadata
  defaults.

The HTTP-server-bring-up + sandbox lifecycle paths in ``_runner.py`` are
exercised by the e2e test (Phase 5/6), not here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# normalize_sample
# ---------------------------------------------------------------------------


def test_normalize_sample_extracts_remote_env_info_fields():
    """A jsonl row carrying ``metadata.remote_env_info`` must produce a dict
    with ``instance_id / image / workdir / pre_commands`` mirrored from it."""
    from tests.test_coding_agent._runner import normalize_sample

    raw = {
        "metadata": {
            "remote_env_info": {
                "instance_id": "django__django-12345",
                "image_url": "image-hub.glm.ai/scaleswe.oh.34:1",
                "workdir": "/workspace/django",
                "pre_commands": ["pip install -r requirements.txt"],
            }
        },
        "prompt": [{"role": "user", "content": "Fix the bug."}],
        "label": "patch goes here",
    }
    out = normalize_sample(raw)
    assert out["instance_id"] == "django__django-12345"
    assert out["image"] == "image-hub.glm.ai/scaleswe.oh.34:1"
    assert out["workdir"] == "/workspace/django"
    assert out["pre_commands"] == ["pip install -r requirements.txt"]
    assert out["problem_statement"] == "Fix the bug."
    assert out["label"] == "patch goes here"


def test_normalize_sample_falls_back_to_unknown_when_no_metadata():
    """Missing/blank metadata must not crash; instance_id defaults to
    'unknown' and image/workdir are None."""
    from tests.test_coding_agent._runner import normalize_sample

    out = normalize_sample({"prompt": [], "label": None})
    assert out["instance_id"] == "unknown"
    assert out["image"] is None
    assert out["workdir"] is None
    assert out["pre_commands"] is None
    assert out["problem_statement"] == ""
    assert out["label"] is None


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_load_dataset_applies_offset_then_limit(tmp_path):
    """offset=2 / limit=3 must skip the first 2 valid rows then return the
    next 3."""
    from tests.test_coding_agent._runner import load_dataset

    p = tmp_path / "data.jsonl"
    rows = [{"metadata": {"remote_env_info": {"instance_id": f"id-{i}"}}} for i in range(8)]
    _write_jsonl(p, rows)
    out = load_dataset(p, offset=2, limit=3)
    assert [r["instance_id"] for r in out] == ["id-2", "id-3", "id-4"]


def test_load_dataset_skips_blank_lines_and_bad_json(tmp_path, capsys):
    """Blank lines and lines that fail JSON parsing must be skipped (with a
    warning on stderr) but not break iteration."""
    from tests.test_coding_agent._runner import load_dataset

    p = tmp_path / "messy.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps({"metadata": {"remote_env_info": {"instance_id": "a"}}}),
                "",
                "{not json",
                json.dumps({"metadata": {"remote_env_info": {"instance_id": "b"}}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = load_dataset(p, offset=0, limit=10)
    assert [r["instance_id"] for r in out] == ["a", "b"]


# ---------------------------------------------------------------------------
# ensure_e2b_env
# ---------------------------------------------------------------------------


def test_ensure_e2b_env_substitutes_dummy_when_key_missing(monkeypatch):
    """Without E2B_API_KEY the helper must inject a syntactically valid
    ``e2b_<40 hex>`` dummy that the E2B SDK's local format check accepts."""
    import re

    from tests.test_coding_agent._runner import ensure_e2b_env

    monkeypatch.delenv("E2B_API_KEY", raising=False)
    ensure_e2b_env()
    val = os.environ.get("E2B_API_KEY", "")
    assert re.fullmatch(r"e2b_[0-9a-fA-F]{40}", val), f"got {val!r}"


def test_ensure_e2b_env_keeps_real_key(monkeypatch):
    """A real-looking key must be preserved untouched."""
    from tests.test_coding_agent._runner import ensure_e2b_env

    real = "e2b_" + "a" * 40
    monkeypatch.setenv("E2B_API_KEY", real)
    ensure_e2b_env()
    assert os.environ["E2B_API_KEY"] == real


def test_ensure_e2b_env_sets_image_metadata_defaults(monkeypatch):
    """The helper must set ``SWE_SANDBOX_IMAGE_METADATA_KEY`` etc. when they
    aren't already in the environment, but not overwrite existing values."""
    from tests.test_coding_agent._runner import ensure_e2b_env

    monkeypatch.delenv("E2B_API_KEY", raising=False)
    monkeypatch.delenv("SWE_SANDBOX_IMAGE_METADATA_KEY", raising=False)
    monkeypatch.setenv("SWE_SANDBOX_METADATA_FILE", "/custom/path")
    ensure_e2b_env()
    assert os.environ["SWE_SANDBOX_IMAGE_METADATA_KEY"] == "glm-platform/image"
    assert os.environ["SWE_SANDBOX_METADATA_FILE"] == "/custom/path"


# ---------------------------------------------------------------------------
# start_adapter_app: in-process aiohttp server lifecycle
# ---------------------------------------------------------------------------


def _make_args(**overrides):
    """Build the small Namespace ``start_adapter_app`` expects. Only fields
    the helper reads need to be present; tests can override per case."""
    import argparse

    defaults = dict(
        model="actor",
        sglang_url="http://127.0.0.1:30000",
        tool_parser="qwen3_coder",
        reasoning_parser="qwen3",
        host_ip="127.0.0.1",
        port=0,  # ask the OS to pick a free port
        max_turns_per_sid=100,
        tito_snapshot_min_loss_tokens=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_start_adapter_app_binds_port_and_serves_healthz(tmp_path):
    """``start_adapter_app(args, batch_dir, tokenizer)`` must return a context
    manager (or async setup result) that exposes a working /healthz on the
    bound port. The adapter must be wired with ``max_turns_per_sid`` from args
    and have the dump layer installed (so ``debug_handle`` is non-None)."""
    import asyncio

    import aiohttp
    from tests.test_coding_agent._runner import start_adapter_app

    async def run_case():
        args = _make_args(port=0, max_turns_per_sid=7)
        # Reuse the toy tokenizer from sibling fixture.
        tok = ScriptedTokenizer(prompts=[], outputs={})
        async with start_adapter_app(args, tmp_path, tok) as session:
            assert session.adapter is not None
            assert session.adapter.max_turns_per_sid == 7
            assert session.debug_handle is not None
            assert session.port > 0
            url = f"http://127.0.0.1:{session.port}/healthz"
            async with aiohttp.ClientSession() as sess:
                async with sess.get(url, timeout=aiohttp.ClientTimeout(total=2)) as r:
                    assert r.status == 200

    asyncio.run(run_case())


def test_start_adapter_app_dumps_per_sid_under_batch_dir(tmp_path):
    """A /v1/messages call through the started app must produce a per-sid
    dump directory under ``batch_dir`` (proves install_dump_layer was wired
    with run_dir=batch_dir)."""
    import asyncio

    import aiohttp
    from aiohttp import web
    from tests.test_coding_agent._runner import start_adapter_app

    async def run_case():
        upstream = FakeSGLang([[(-0.1, 200), (-0.2, 201)]])
        upstream_app = web.Application()
        upstream_app.router.add_post("/generate", upstream.handle_generate)
        upstream_server = TestServer(upstream_app)
        await upstream_server.start_server()
        try:
            args = _make_args(
                port=0,
                sglang_url=str(upstream_server.make_url("")).rstrip("/"),
            )
            tok = ScriptedTokenizer(prompts=[[1, 2]], outputs={(200, 201): "hi"})
            async with start_adapter_app(args, tmp_path, tok) as session:
                payload = {
                    "model": "actor",
                    "max_tokens": 5,
                    "messages": [{"role": "user", "content": "hi"}],
                }
                headers = {"Authorization": "Bearer sid-runner"}
                async with aiohttp.ClientSession() as sess:
                    url = f"http://127.0.0.1:{session.port}/v1/messages"
                    async with sess.post(url, json=payload, headers=headers) as r:
                        assert r.status == 200
                # The dump layer must have written per-sid artefacts under
                # batch_dir/<sid>/turn_NNNN_*.json.
                sid_dir = tmp_path / "sid-runner"
                assert sid_dir.is_dir()
                assert (sid_dir / "turn_0001_request.json").is_file()
        finally:
            await upstream_server.close()

    asyncio.run(run_case())


# ---------------------------------------------------------------------------
# drain_and_dump_sid: in-process replacement for the old /get_trajectory +
# inst_dir reconciliation that lived at the tail of run_one_instance.
# ---------------------------------------------------------------------------


def test_drain_and_dump_sid_writes_tree_trajectory_and_extracts_stats(tmp_path):
    """After a sid has had at least one turn appended, calling
    ``drain_and_dump_sid(adapter, debug_handle, sid, inst_dir, sample)``
    must:

      * write ``inst_dir/trajectory_tree.{txt,json}`` (via on_drain_start)
      * write ``inst_dir/trajectory.json`` (via on_drain_done)
      * return a partial-summary dict carrying ``tree`` (compute_tree_stats
        output) + ``tito_dropped_tokens`` + ``tito_dropped_turns`` +
        ``num_samples``
    """
    import asyncio

    import aiohttp
    from aiohttp import web
    from tests.test_coding_agent._runner import drain_and_dump_sid, start_adapter_app

    async def run_case():
        upstream = FakeSGLang(
            [
                [(-0.1, 200), (-0.2, 201)],
            ]
        )
        upstream_app = web.Application()
        upstream_app.router.add_post("/generate", upstream.handle_generate)
        upstream_server = TestServer(upstream_app)
        await upstream_server.start_server()
        try:
            args = _make_args(
                port=0,
                sglang_url=str(upstream_server.make_url("")).rstrip("/"),
            )
            tok = ScriptedTokenizer(prompts=[[1, 2]], outputs={(200, 201): "hi"})
            async with start_adapter_app(args, tmp_path, tok) as session:
                # Send a single /v1/messages turn so the sid has a tree.
                sid = "sid-drain"
                async with aiohttp.ClientSession() as sess:
                    url = f"http://127.0.0.1:{session.port}/v1/messages"
                    headers = {"Authorization": f"Bearer {sid}"}
                    payload = {
                        "model": "actor",
                        "max_tokens": 5,
                        "messages": [{"role": "user", "content": "hi"}],
                    }
                    async with sess.post(url, json=payload, headers=headers) as r:
                        assert r.status == 200

                inst_dir = tmp_path / "inst_0001"
                inst_dir.mkdir()
                # The drain step routes outputs to inst_dir (override
                # sid_dump_dir before draining).
                session.debug_handle.sid_dump_dir[sid] = str(inst_dir)
                partial = await drain_and_dump_sid(
                    adapter=session.adapter,
                    debug_handle=session.debug_handle,
                    sid=sid,
                    inst_dir=inst_dir,
                    sample={"label": "patch", "problem_statement": "fix it"},
                )

                assert (inst_dir / "trajectory_tree.json").is_file()
                assert (inst_dir / "trajectory_tree.txt").is_file()
                assert (inst_dir / "trajectory.json").is_file()
                assert partial["tree"]["found"] is True
                assert partial["tree"]["turns"] == 1
                assert partial["num_samples"] >= 1
                assert "tito_dropped_tokens" in partial
                assert "tito_dropped_turns" in partial
        finally:
            await upstream_server.close()

    asyncio.run(run_case())


# Sibling fixture imports for the start_adapter_app tests above.
import sys as _sys
from pathlib import Path as _Path

_THIS_DIR = _Path(__file__).resolve().parent
_sys.path.insert(0, str(_THIS_DIR.parent))
from aiohttp.test_utils import TestServer  # noqa: E402
from test_agent_adapters import FakeSGLang, ScriptedTokenizer  # noqa: E402
