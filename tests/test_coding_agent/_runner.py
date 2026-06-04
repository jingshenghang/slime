"""E2E test runner: in-process AnthropicAdapter + E2B sandbox orchestration.

This module owns the pieces of the historic ``launch_swe.py`` that DO NOT
relate to debug dumping or summary aggregation:

* dataset loading + per-row normalisation
* E2B environment sanity-check
* in-process AnthropicAdapter app lifecycle (start_adapter_app /
  shutdown_adapter_app) — exposes the same /v1/messages endpoint a stand-
  alone middleware process would, but without the subprocess
* per-instance sandbox worker (run_one_instance)
* batch driver (amain) that fans out workers under an asyncio.Semaphore

Debug-dump hooks come from :mod:`_dump_helpers`; per-batch aggregation comes
from :mod:`_analysis`.

The CLI surface (argparse, ``--limit``, ``--runs-dir`` etc.) lives in
``test_coding_agent_swe_e2e.py``; this module exposes pure-ish helpers so
unit tests can exercise normalize_sample / load_dataset / ensure_e2b_env
without spinning up the full e2e stack.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Default sandbox metadata file (GLM platform requires a specific shape; see
# memory/e2b/glm_e2b_metadata_restrictions.md). Override by setting
# ``SWE_SANDBOX_METADATA_FILE`` in the environment.
_DEFAULT_SANDBOX_METADATA_FILE = "/mnt/jingshenghang/code/slime_swe/0521/configs/sandbox_metadata.example.json"


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------


def normalize_sample(raw: dict[str, Any]) -> dict[str, Any]:
    """Project a slime-SWE jsonl row to the small dict the worker consumes.

    Pulls ``instance_id / image_url / workdir / pre_commands`` from
    ``raw['metadata']['remote_env_info']``; falls back to ``unknown`` / None
    when the metadata is missing. The first user prompt becomes
    ``problem_statement`` (passed to the sandbox as PROBLEM_STATEMENT.md).
    """
    rem = (raw.get("metadata") or {}).get("remote_env_info") or {}
    prompt_obj = raw.get("prompt") or []
    if isinstance(prompt_obj, list) and prompt_obj:
        first = prompt_obj[0]
        problem_statement = first.get("content", "") if isinstance(first, dict) else ""
    else:
        problem_statement = str(prompt_obj) if prompt_obj else ""
    return {
        "instance_id": rem.get("instance_id") or "unknown",
        "image": rem.get("image_url"),
        "workdir": rem.get("workdir"),
        "pre_commands": rem.get("pre_commands"),
        "problem_statement": problem_statement,
        "label": raw.get("label"),
    }


def load_dataset(path: Path | str, offset: int, limit: int) -> list[dict[str, Any]]:
    """Read at most ``limit`` rows from ``path`` (jsonl), skipping the first
    ``offset`` valid rows. Blank lines and bad-JSON lines are skipped with a
    warning on stderr."""
    path = Path(path)
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        skipped = 0
        for i, raw_line in enumerate(f):
            if len(out) >= limit:
                break
            line = raw_line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[load] skipping line {i}: bad json: {e}", file=sys.stderr)
                continue
            if skipped < offset:
                skipped += 1
                continue
            out.append(normalize_sample(raw))
    return out


# ---------------------------------------------------------------------------
# E2B environment
# ---------------------------------------------------------------------------


def ensure_e2b_env() -> None:
    """Inject a syntactically-valid placeholder E2B_API_KEY when the real one
    is missing/malformed, plus standard sandbox metadata defaults.

    The official E2B SDK validates the key format locally (matches
    ``^e2b_[0-9a-f]{40}$``). GLM's internal gateway ignores the value but the
    SDK still rejects garbage; the dummy lets the SDK construct.
    """
    key = os.environ.get("E2B_API_KEY", "")
    if not re.fullmatch(r"e2b_[0-9a-fA-F]{40}", key):
        os.environ["E2B_API_KEY"] = "e2b_" + "0" * 40
    defaults = {
        "SWE_SANDBOX_METADATA_FILE": _DEFAULT_SANDBOX_METADATA_FILE,
        "SWE_SANDBOX_IMAGE_METADATA_KEY": "glm-platform/image",
    }
    for k, v in defaults.items():
        os.environ.setdefault(k, v)


__all__ = [
    "normalize_sample",
    "load_dataset",
    "ensure_e2b_env",
    "start_adapter_app",
    "AdapterSession",
    "drain_and_dump_sid",
    "run_one_instance",
    "amain",
]


# ---------------------------------------------------------------------------
# In-process adapter app lifecycle
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class AdapterSession:
    """Resources owned while the adapter HTTP app is running.

    Returned from :func:`start_adapter_app` (used as an async context manager).
    Fields are the things ``run_one_instance`` needs to talk to the adapter
    and write per-sid dumps without an extra subprocess hop.
    """

    adapter: Any  # AnthropicAdapter
    debug_handle: Any  # _dump_helpers.Debug
    port: int
    runner: Any  # aiohttp.web.AppRunner — kept so we can clean it up


@contextlib.asynccontextmanager
async def start_adapter_app(args, batch_dir: Path, tokenizer):
    """Bring up an in-process :class:`AnthropicAdapter` HTTP app on
    ``args.host_ip:args.port`` and yield an :class:`AdapterSession`.

    On enter:
      * constructs the adapter with ``max_turns_per_sid`` and
        ``tito_snapshot_min_loss_tokens`` from ``args``
      * installs the dump layer so per-sid turn artefacts land under
        ``batch_dir/<sid>/`` (or wherever ``sid_dump_dir`` later overrides)
      * binds ``aiohttp.web.AppRunner`` on the requested port (use ``0`` to
        ask the OS for a free port; the actual port is exposed via
        ``session.port``)

    On exit the AppRunner is cleaned up. The adapter object survives so a
    final drain can still run after the server stops serving.
    """
    from aiohttp import web
    from tests.test_coding_agent._dump_helpers import install_dump_layer

    from slime.agent.adapters.anthropic import AnthropicAdapter

    batch_dir = Path(batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)

    adapter = AnthropicAdapter(
        tokenizer=tokenizer,
        sglang_url=args.sglang_url.rstrip("/"),
        tool_parser=getattr(args, "tool_parser", None),
        reasoning_parser=getattr(args, "reasoning_parser", None),
        tito_snapshot_min_loss_tokens=getattr(args, "tito_snapshot_min_loss_tokens", None),
        max_turns_per_sid=getattr(args, "max_turns_per_sid", None),
    )
    debug_handle = install_dump_layer(
        adapter,
        run_dir=batch_dir,
        tokenizer=tokenizer,
        sid_dump_dir={},
    )

    runner = web.AppRunner(adapter.app)
    await runner.setup()
    # Bind 0.0.0.0 (not args.host_ip): some hosts expose the dial-back IP via
    # kube-ipvs0 VIPs that can't be `listen()`-ed directly. Sandboxes still
    # dial args.host_ip; reachability is verified per-instance.
    site = web.TCPSite(runner, "0.0.0.0", args.port)
    await site.start()
    # Resolve the OS-assigned port when args.port == 0.
    actual_port = args.port
    if actual_port == 0:
        socks = list(site._server.sockets) if site._server is not None else []
        if socks:
            actual_port = socks[0].getsockname()[1]

    session = AdapterSession(
        adapter=adapter,
        debug_handle=debug_handle,
        port=actual_port,
        runner=runner,
    )
    try:
        yield session
    finally:
        await runner.cleanup()


# ---------------------------------------------------------------------------
# drain_and_dump_sid: in-process replacement for /get_trajectory + the
# inst_dir reconciliation that lived at the tail of launch_swe.run_one_instance.
# ---------------------------------------------------------------------------


async def drain_and_dump_sid(
    *,
    adapter,
    debug_handle,
    sid: str,
    inst_dir: Path,
    sample: dict[str, Any],
) -> dict[str, Any]:
    """Drain a session, write its tree + trajectory artefacts, return summary.

    Sequence (matches the historic middleware ``/get_trajectory`` route +
    launch_swe inst_dir reconciliation, but in-process):

      1. ``debug_handle.on_drain_start(sid, adapter.manager)`` writes
         ``inst_dir/trajectory_tree.{json,txt}`` snapshots of the live tree
         BEFORE drainage pops it.
      2. ``adapter.finish_session(sid, ...)`` linearises the tree to a list
         of :class:`Sample` and pops the sid.
      3. ``debug_handle.on_drain_done(sid, samples)`` writes
         ``inst_dir/trajectory.json`` with decoded prompt/response text.
      4. Compute ``tree`` stats, sum per-sample TITO drops, return partial
         summary that the caller merges into the worker's full summary.
    """
    from tests.test_coding_agent._analysis import compute_tree_stats

    inst_dir = Path(inst_dir)
    inst_dir.mkdir(parents=True, exist_ok=True)
    # Make sure later hooks land in inst_dir even if the caller forgot to
    # route the sid earlier.
    debug_handle.sid_dump_dir[sid] = str(inst_dir)

    # 1) snapshot tree before drain pops it
    try:
        debug_handle.on_drain_start(sid, adapter.manager)
    except Exception as e:
        print(f"[drain] on_drain_start(sid={sid}) failed: {e}", file=sys.stderr)

    # 2) drain
    samples = await adapter.finish_session(
        sid,
        reward=0.0,
        extra_metadata={"label": sample.get("label")} if sample.get("label") is not None else None,
    )

    # 3) write trajectory.json
    try:
        debug_handle.on_drain_done(sid, samples)
    except Exception as e:
        print(f"[drain] on_drain_done(sid={sid}) failed: {e}", file=sys.stderr)

    # 4) compute stats from the tree json we just wrote (round-trip so we
    # exercise the same on-disk contract the analyzer reads)
    partial: dict[str, Any] = {
        "num_samples": len(samples),
        "tito_dropped_tokens": 0,
        "tito_dropped_turns": 0,
        # Snapshot Sample(s) emitted when a TITO drift would otherwise drop
        # >= tito_snapshot_min_loss_tokens loss tokens. Each snapshot carries
        # the dropped suffix as a complementary loss_mask so the tokens still
        # contribute to training; counted here for summary reporting.
        "tito_snapshots_count": 0,
        "tito_snapshot_loss_tokens": 0,
        "tito_snapshot_turns": [],
        "tree": None,
    }
    for s in samples:
        md = getattr(s, "metadata", None) or {}
        partial["tito_dropped_tokens"] += int(md.get("tito_dropped_tokens", 0) or 0)
        partial["tito_dropped_turns"] += int(md.get("tito_dropped_turns", 0) or 0)
        if md.get("tito_snapshot"):
            partial["tito_snapshots_count"] += 1
            partial["tito_snapshot_loss_tokens"] += int(md.get("tito_snapshot_loss_tokens", 0) or 0)
            at_turn = md.get("tito_snapshot_at_turn")
            if at_turn is not None:
                partial["tito_snapshot_turns"].append(int(at_turn))

    tree_json_path = inst_dir / "trajectory_tree.json"
    if tree_json_path.exists():
        try:
            tree = json.loads(tree_json_path.read_text())
            partial["tree"] = compute_tree_stats(tree)
        except Exception as e:
            partial["tree"] = {"error": f"parse: {e}"}
    else:
        partial["tree"] = {"error": "no_trajectory_tree_json"}

    return partial


# ---------------------------------------------------------------------------
# Sandbox-coupled e2e runner
# ---------------------------------------------------------------------------
#
# These pieces drive a real E2B sandbox and the host-side claude-code CLI,
# so unit-tests can't run them. Behavioural verification happens in the
# end-to-end smoke (Step 2 of the plan's verification section) and in the
# large-batch run (Step 3). They mirror the historic ``launch_swe.py``
# worker + main loop, with three deltas:
#   * no middleware subprocess — adapter app is in-process
#   * no /register_sid HTTP call — we write the override into
#     ``debug_handle.sid_dump_dir`` directly
#   * drain via ``drain_and_dump_sid``, not /get_trajectory
# ---------------------------------------------------------------------------


_DEFAULT_NODE_TGZ = Path(
    os.environ.get(
        "SWE_HOST_NODE_TARBALL",
        "/mnt/jingshenghang/software/node-v22.20.0-linux-x64.tar.xz",
    )
)
_DEFAULT_CC_TGZ = Path(
    os.environ.get(
        "SWE_HOST_CC_TARBALL",
        "/mnt/jingshenghang/storage/claude_code/anthropic-ai-claude-code-2.1.143-local-linux-x64.tgz",
    )
)
DEFAULT_DATASET = Path("/mnt/jingshenghang/storage/datasets/ca/data/swe-train-1545-slime.jsonl")
DEFAULT_MODEL = "/mnt/jingshenghang/storage/checkpoints/Qwen3.6-35B-A3B"
DEFAULT_RUNS_DIR = Path("/mnt/jingshenghang/code/slime_swe/0603-trajectory-manager/runs/swe_new")
SWE_CC_PROMPT = (
    "Read PROBLEM_STATEMENT.md in the current directory and resolve the issue. "
    "Edit source files only (do NOT touch tests). After editing, run the relevant "
    "tests to verify your fix passes. Do NOT modify PROBLEM_STATEMENT.md and do "
    "NOT commit. When finished, print a one-line summary and exit."
)
NODE_ENV = {
    "PATH": "/opt/node22/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "NODE_PATH": "/opt/node22/lib/node_modules",
}


async def _install_node_and_cc(sb, args, log_prefix: str) -> None:
    if not args.node_tgz.exists():
        raise FileNotFoundError(f"--node-tgz not found: {args.node_tgz}")
    if not args.cc_tgz.exists():
        raise FileNotFoundError(f"--cc-tgz not found: {args.cc_tgz}")
    await sb.write_file("/tmp/node.tar.xz", args.node_tgz)
    await sb.write_file("/tmp/cc.tgz", args.cc_tgz)
    await sb.exec(
        "mkdir -p /opt/node22 && tar -xJf /tmp/node.tar.xz -C /opt/node22 --strip-components=1",
        timeout=180,
        check=True,
    )
    await sb.exec(
        "ln -sf /opt/node22/bin/node /usr/local/bin/node && "
        "ln -sf /opt/node22/bin/npm /usr/local/bin/npm && "
        "ln -sf /opt/node22/bin/npx /usr/local/bin/npx",
        timeout=30,
        check=True,
    )
    await sb.exec(
        "/opt/node22/bin/npm install -g --prefix=/usr/local /tmp/cc.tgz",
        env=NODE_ENV,
        timeout=600,
        check=True,
    )
    rc, out, _ = await sb.exec("claude --version", env=NODE_ENV, timeout=30, check=True)
    print(f"[{log_prefix}] claude --version: {out.strip()}")


async def _bootstrap_agent_user(sb, workdir: str) -> None:
    script = (
        "set -e; "
        "id agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent; "
        "mkdir -p /home/agent/.claude " + workdir + "; "
        'printf %s \'{"hasCompletedOnboarding": true, '
        '"bypassPermissionsModeAccepted": true}\' '
        "| tee /home/agent/.claude.json /home/agent/.claude/settings.json "
        ">/dev/null; "
        "chown -R agent:agent /home/agent " + workdir + "; "
        "git config --system --add safe.directory '*'"
    )
    await sb.exec(script, timeout=60, check=True)


async def _apply_pre_commands(sb, workdir: str, pre) -> None:
    import shlex as _shlex

    if not pre:
        return
    if isinstance(pre, str):
        body = pre.replace("\\n", "\n")
    else:
        body = "\n".join(c for c in pre if c)
    await sb.write_file("/workspace/__cagent_pre__.sh", "set -e\n" + body, user="agent")
    await sb.exec(
        f"chmod 755 /workspace/__cagent_pre__.sh && "
        f"cd {_shlex.quote(workdir)} && bash /workspace/__cagent_pre__.sh",
        user="agent",
        check=False,
        timeout=600,
    )


async def _reachability_check(sb, host_ip: str, port: int) -> None:
    import shlex as _shlex

    url = f"http://{host_ip}:{port}/healthz"
    rc, out, err = await sb.exec(
        f"curl -sS --max-time 5 -o /dev/null -w '%{{http_code}}' {_shlex.quote(url)}",
        env=NODE_ENV,
        timeout=15,
    )
    code = out.strip()
    if rc != 0 or code != "200":
        raise RuntimeError(
            f"sandbox cannot reach host adapter at {url} " f"(curl rc={rc}, http={code!r}, err={err[:200]!r})"
        )


async def _run_claude(sb, args, *, sid: str, workdir: str, model: str, inst_dir: Path) -> int:
    import shlex as _shlex

    env = {
        **NODE_ENV,
        "HOME": "/home/agent",
        "ANTHROPIC_BASE_URL": f"http://{args.host_ip}:{args.port}",
        "ANTHROPIC_AUTH_TOKEN": sid,
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_API_KEY": "dummy",
    }
    inner = (
        f"cd {_shlex.quote(workdir)} && "
        f"claude -p {_shlex.quote(args.cc_prompt)} "
        f"--permission-mode bypassPermissions"
    )
    cmd = f"runuser -u agent -- bash -c {_shlex.quote(inner)} < /dev/null"
    rc, out, err = await sb.exec(cmd, env=env, timeout=args.claude_timeout)
    (inst_dir / "stdout.log").write_text(out, encoding="utf-8")
    (inst_dir / "stderr.log").write_text(err, encoding="utf-8")
    return rc


async def run_one_instance(
    args,
    batch_dir: Path,
    idx: int,
    sample: dict[str, Any],
    session: AdapterSession,
) -> dict[str, Any]:
    """Run one SWE instance against the in-process adapter on
    ``session.port``. Mirrors launch_swe.run_one_instance but routes
    dump_dir + drain in-process instead of through HTTP."""
    import time
    import traceback

    instance_id = sample["instance_id"]
    safe_id = instance_id.replace("/", "_")[:80]
    inst_dir = batch_dir / f"{idx:04d}_{safe_id}"
    inst_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "idx": idx,
        "instance_id": instance_id,
        "image": sample["image"],
        "workdir": sample["workdir"],
        "inst_dir": str(inst_dir),
        "sid": None,
        "sandbox_id": None,
        "rc": None,
        "elapsed_sec": None,
        "tree": None,
        "error": None,
    }

    if not sample["image"] or not sample["workdir"]:
        summary["error"] = "missing_image_or_workdir"
        (inst_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    ts = time.strftime("%Y%m%d-%H%M%S")
    sid = f"swe-{idx:04d}-{ts}"
    summary["sid"] = sid

    meta = {
        "ts": ts,
        "idx": idx,
        "instance_id": instance_id,
        "image": sample["image"],
        "workdir": sample["workdir"],
        "label_len": len(sample["label"] or ""),
        "problem_statement_len": len(sample["problem_statement"]),
        "sid": sid,
        "model": args.model,
        "host_ip": args.host_ip,
        "port": args.port,
        "max_turns_per_sid": args.max_turns_per_sid,
    }
    (inst_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    (inst_dir / "PROBLEM_STATEMENT.md").write_text(sample["problem_statement"], encoding="utf-8")

    # Pre-register so cc's first /v1/messages dumps already land in inst_dir.
    session.debug_handle.sid_dump_dir[sid] = str(inst_dir)
    print(f"[{idx:04d}] {instance_id}: sid={sid} starting (image={sample['image']})")

    t0 = time.time()
    try:
        from slime.agent.sandbox import E2BSandbox  # type: ignore

        async with E2BSandbox(image=sample["image"], timeout=args.sandbox_timeout) as sb:
            summary["sandbox_id"] = sb.sandbox_id
            (inst_dir / "meta.json").write_text(
                json.dumps({**meta, "sandbox_id": sb.sandbox_id}, indent=2, default=str),
                encoding="utf-8",
            )
            await _install_node_and_cc(sb, args, log_prefix=f"{idx:04d}")
            await _bootstrap_agent_user(sb, sample["workdir"])
            await _apply_pre_commands(sb, sample["workdir"], sample["pre_commands"])
            await sb.write_file(
                f"{sample['workdir']}/PROBLEM_STATEMENT.md",
                sample["problem_statement"] or "",
                user="agent",
            )
            await _reachability_check(sb, args.host_ip, args.port)
            rc = await _run_claude(
                sb,
                args,
                sid=sid,
                workdir=sample["workdir"],
                model=args.model,
                inst_dir=inst_dir,
            )
            summary["rc"] = rc
    except Exception as e:
        summary["error"] = f"{type(e).__name__}: {e}"
        (inst_dir / "exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"[{idx:04d}] FAILED during sandbox phase: {e!r}", file=sys.stderr)
    finally:
        try:
            partial = await drain_and_dump_sid(
                adapter=session.adapter,
                debug_handle=session.debug_handle,
                sid=sid,
                inst_dir=inst_dir,
                sample=sample,
            )
            summary.update(partial)
        except Exception as e:
            print(f"[{idx:04d}] drain failed: {e!r}", file=sys.stderr)
        summary["elapsed_sec"] = round(time.time() - t0, 1)

    (inst_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    t = summary.get("tree") or {}
    print(
        f"[{idx:04d}] DONE rc={summary.get('rc')} "
        f"turns={t.get('turns', '-')} leaves={t.get('leaves', '-')} "
        f"forks={t.get('n_forks', '-')} "
        f"tito_drop={summary.get('tito_dropped_turns', 0)}t/"
        f"{summary.get('tito_dropped_tokens', 0)}k "
        f"elapsed={summary['elapsed_sec']}s"
    )
    return summary


async def amain(args) -> int:
    """Batch driver: load dataset, bring up adapter app, fan out workers,
    aggregate summary."""
    import asyncio
    import time
    import traceback

    from tests.test_coding_agent._analysis import write_summary
    from transformers import AutoTokenizer

    ensure_e2b_env()
    ts = time.strftime("%Y%m%d-%H%M%S")
    batch_dir = (Path(args.runs_dir) / ts).resolve()
    batch_dir.mkdir(parents=True)
    print(f"[swe_e2e] batch_dir = {batch_dir}")

    rows = load_dataset(args.dataset, args.offset, args.limit)
    print(f"[swe_e2e] loaded {len(rows)} rows " f"(offset={args.offset}, limit={args.limit})")

    (batch_dir / "batch_meta.json").write_text(
        json.dumps(
            {
                "ts": ts,
                "cmd": sys.argv,
                "dataset": str(args.dataset),
                "offset": args.offset,
                "limit": args.limit,
                "concurrency": args.concurrency,
                "max_turns_per_sid": args.max_turns_per_sid,
                "n_rows": len(rows),
                "model": args.model,
                "host_ip": args.host_ip,
                "sglang_url": args.sglang_url,
                "tool_parser": args.tool_parser,
                "reasoning_parser": args.reasoning_parser,
                "port": args.port,
                "tito_snapshot_min_loss_tokens": getattr(args, "tito_snapshot_min_loss_tokens", None),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(f"[swe_e2e] loading tokenizer from {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    results: list[dict] = []

    async with start_adapter_app(args, batch_dir, tok) as session:
        print(f"[swe_e2e] adapter app serving on {args.host_ip}:{session.port}")
        sem = asyncio.Semaphore(args.concurrency)

        async def worker(i: int, sample: dict[str, Any]) -> None:
            async with sem:
                try:
                    r = await run_one_instance(args, batch_dir, i, sample, session)
                except Exception as e:
                    r = {
                        "idx": i,
                        "instance_id": sample.get("instance_id"),
                        "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc(),
                    }
                    print(
                        f"[{i:04d}] unhandled worker exception: {e!r}",
                        file=sys.stderr,
                    )
                results.append(r)
                # Incremental summary so progress is inspectable mid-run.
                try:
                    write_summary(batch_dir, sorted(results, key=lambda r: r.get("idx", 0)))
                except Exception:
                    pass

        await asyncio.gather(*(worker(i + args.offset, s) for i, s in enumerate(rows)))

    results.sort(key=lambda r: r.get("idx", 0))
    write_summary(batch_dir, results)
    print(f"[swe_e2e] done. {len(results)} instances. " f"summary at {batch_dir / 'summary.txt'}")
    return 0
