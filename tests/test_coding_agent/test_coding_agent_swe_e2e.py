"""End-to-end test for the slime coding agent rollout (SWE benchmark flavor).

This is the post-refactor replacement for the historic
``examples/coding_agent_rl/launch_swe.py``: it boots an in-process
:class:`~slime.agent.adapters.anthropic.AnthropicAdapter`, fans out N
E2B sandboxes (each running ``claude -p ...`` against the adapter), and
writes the same on-disk dump layout under ``--runs-dir`` so the trees can
be compared against historic ``runs/swe/`` baselines.

Two entry points:

  * **CLI (large batches)**::

        python -m tests.test_coding_agent.test_coding_agent_swe_e2e \\
            --limit 100 --concurrency 16

  * **pytest smoke (CI-friendly)**::

        pytest tests/test_coding_agent/test_coding_agent_swe_e2e.py -k smoke

The smoke test is opt-in (skipped unless ``SWE_E2E_SMOKE=1``) so an offline
``pytest tests/test_coding_agent/`` run stays fast.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from tests.test_coding_agent import _runner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="test_coding_agent_swe_e2e")
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
        help="Adapter caps each sid at this many /v1/messages turns (HTTP 429).",
    )
    p.add_argument("--model", default=_runner.DEFAULT_MODEL)
    p.add_argument("--tool-parser", default="qwen3_coder")
    p.add_argument("--reasoning-parser", default="qwen3")
    p.add_argument("--sglang-url", default="http://127.0.0.1:30000")
    p.add_argument(
        "--host-ip",
        default=os.environ.get("SLIME_HEAD_HOST", "172.27.14.123"),
        help="IP the sandboxes dial back to. Default is the dev-machine VIP; "
        "override via $SLIME_HEAD_HOST or --host-ip on other nodes.",
    )
    p.add_argument(
        "--port",
        type=int,
        default=18080,
        help="Adapter app listen port (sandbox->host reverse). 18080 is the "
        "platform-open port; pass 0 to let the OS pick (useful for tests).",
    )
    p.add_argument(
        "--node-tgz",
        type=Path,
        default=_runner._DEFAULT_NODE_TGZ,
        help="Host-side Node.js 22 tarball (uploaded into the sandbox).",
    )
    p.add_argument(
        "--cc-tgz",
        type=Path,
        default=_runner._DEFAULT_CC_TGZ,
        help="Host-side claude-code tgz (npm install -g inside the sandbox).",
    )
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=_runner.DEFAULT_RUNS_DIR,
        help="Where per-batch run dirs land. Default writes to " "0603-trajectory-manager/runs/swe_new/.",
    )
    p.add_argument(
        "--sandbox-timeout",
        type=int,
        default=1800,
        help="E2B sandbox total lifetime in seconds.",
    )
    p.add_argument(
        "--claude-timeout",
        type=int,
        default=1500,
        help="Per-instance budget for the claude-code call. Should be < " "sandbox-timeout to leave room for cleanup.",
    )
    p.add_argument(
        "--cc-prompt",
        default=_runner.SWE_CC_PROMPT,
        help="Driver prompt for `claude -p`; the real task lives in PROBLEM_STATEMENT.md.",
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
    """Tiny end-to-end with --limit 2; opt-in via SWE_E2E_SMOKE=1.

    The smoke needs E2B + an sglang upstream + the model checkpoint; we skip
    by default so ``pytest tests/test_coding_agent/`` stays offline-clean.
    Set ``SWE_E2E_SMOKE=1`` (plus any env vars the run needs) to enable.
    """
    import pytest

    if os.environ.get("SWE_E2E_SMOKE") != "1":
        pytest.skip("Set SWE_E2E_SMOKE=1 to run the SWE e2e smoke test.")
    args = parse_args(["--limit", "2", "--concurrency", "2"])
    rc = asyncio.run(_runner.amain(args))
    assert rc == 0


if __name__ == "__main__":
    main()
