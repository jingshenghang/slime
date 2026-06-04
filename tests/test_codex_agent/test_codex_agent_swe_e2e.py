"""End-to-end test for the Codex CLI flavor of the slime coding-agent rollout.

Mirrors ``tests/test_coding_agent/test_coding_agent_swe_e2e.py`` but drives
the OpenAI Chat-Completions adapter + Codex CLI inside an E2B sandbox.

Two entry points:

  * **CLI (large batches)**::

        python -m tests.test_codex_agent.test_codex_agent_swe_e2e \\
            --limit 100 --concurrency 16

  * **pytest smoke (CI-friendly)**::

        SWE_E2E_SMOKE=1 pytest tests/test_codex_agent/test_codex_agent_swe_e2e.py -k smoke

The smoke test is opt-in (skipped unless ``SWE_E2E_SMOKE=1``).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from tests.test_codex_agent import _runner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="test_codex_agent_swe_e2e")
    p.add_argument(
        "--dataset",
        type=Path,
        default=_runner.DEFAULT_DATASET,
        help="JSONL with slime SWE rows (label/prompt/metadata.remote_env_info).",
    )
    p.add_argument("--limit", type=int, default=20, help="Process only the first N rows.")
    p.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N rows before applying --limit.",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Number of instances running in parallel against the shared " "in-process adapter app.",
    )
    p.add_argument(
        "--max-turns-per-sid",
        type=int,
        default=100,
        help="Adapter caps each sid at this many /v1/chat/completions turns (HTTP 429).",
    )
    p.add_argument("--model", default=_runner.DEFAULT_MODEL)
    p.add_argument("--tool-parser", default="qwen3_coder")
    p.add_argument("--reasoning-parser", default="qwen3")
    p.add_argument("--sglang-url", default="http://127.0.0.1:30000")
    p.add_argument(
        "--host-ip",
        default=os.environ.get("SLIME_HEAD_HOST", "172.27.14.123"),
        help="IP the sandboxes dial back to. Default is the dev-machine VIP.",
    )
    p.add_argument(
        "--port",
        type=int,
        default=18001,
        help="Adapter app listen port (sandbox->host reverse). 18001 is the "
        "platform-open port for codex; pass 0 to let the OS pick.",
    )
    p.add_argument(
        "--node-tgz",
        type=Path,
        default=_runner._DEFAULT_NODE_TGZ,
        help="Host-side Node.js 22 tarball (uploaded into the sandbox).",
    )
    p.add_argument(
        "--codex-tgz",
        type=Path,
        default=_runner._DEFAULT_CODEX_TGZ,
        help="Host-side @openai/codex tgz (npm install -g inside the sandbox). "
        "Generate via ``npm pack @openai/codex@0.30.0`` -- 0.30.0 is the last "
        'release that still supports ``wire_api = "chat"``.',
    )
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=_runner.DEFAULT_RUNS_DIR,
        help="Where per-batch run dirs land. Default writes to " "0603-trajectory-manager/runs/swe_codex/.",
    )
    p.add_argument(
        "--sandbox-timeout",
        type=int,
        default=1800,
        help="E2B sandbox total lifetime in seconds.",
    )
    p.add_argument(
        "--codex-timeout",
        type=int,
        default=1500,
        help="Per-instance budget for the codex CLI call. Should be < " "sandbox-timeout to leave room for cleanup.",
    )
    p.add_argument(
        "--codex-prompt",
        default=_runner.SWE_CODEX_PROMPT,
        help="Driver prompt for `codex exec`; the real task lives in PROBLEM_STATEMENT.md.",
    )
    p.add_argument(
        "--tito-snapshot-min-loss-tokens",
        type=int,
        default=1024,
        help="If a TITO drift would drop >= this many loss_mask=1 tokens, "
        "emit an extra snapshot Sample. Pass 0 or a negative value to disable.",
    )
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        rc = asyncio.run(_runner.amain(args))
    except KeyboardInterrupt:
        rc = 130
    sys.exit(rc)


# ---------------------------------------------------------------------------
# pytest smoke
# ---------------------------------------------------------------------------


def test_smoke_e2e():
    """Tiny end-to-end with --limit 1; opt-in via SWE_E2E_SMOKE=1.

    The smoke needs E2B + an sglang upstream + the model checkpoint + a local
    codex tarball; we skip by default so ``pytest tests/test_codex_agent/``
    stays offline-clean. Set ``SWE_E2E_SMOKE=1`` to enable.
    """
    import pytest

    if os.environ.get("SWE_E2E_SMOKE") != "1":
        pytest.skip("Set SWE_E2E_SMOKE=1 to run the SWE codex e2e smoke test.")
    args = parse_args(["--limit", "1", "--concurrency", "1"])
    if not args.codex_tgz.exists():
        pytest.skip(f"codex tarball not found at {args.codex_tgz}; " "pass --codex-tgz or set SWE_HOST_CODEX_TARBALL")
    rc = asyncio.run(_runner.amain(args))
    assert rc == 0


if __name__ == "__main__":
    main()
