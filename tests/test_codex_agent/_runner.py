"""E2E test runner: in-process OpenAIAdapter + E2B sandbox orchestration.

Mirrors ``tests/test_coding_agent/_runner.py`` (anthropic / claude-code) but
drives the Codex CLI inside an E2B sandbox through the OpenAI
Chat-Completions adapter.

The structure is intentionally identical so the dump layouts can be compared
side-by-side; the deltas are:

  * adapter is :class:`slime.agent.adapters.openai.OpenAIAdapter`
  * sandbox bootstrap installs Node 22 then ``npm install -g <codex tarball>``
  * a ``~/.codex/config.toml`` is dropped via base64 round-trip before launch
    (provider ``slime`` with inline ``base_url``, ``wire_api = "chat"``)
  * codex is invoked as ``codex exec --skip-git-repo-check "$PROMPT"`` with
    ``OPENAI_API_KEY=<sid>`` so the adapter resolves the session via
    ``Authorization: Bearer <sid>``
  * default dump root is ``runs/swe_codex/`` and default port is ``18001``
    (the only port the user has open at 172.27.14.123)
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
    """Project a slime-SWE jsonl row to the small dict the worker consumes."""
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
    """Read at most ``limit`` rows from ``path`` (jsonl), skipping ``offset`` rows."""
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
    """Inject placeholder E2B_API_KEY when missing, plus sandbox metadata defaults."""
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
    "DEFAULT_DATASET",
    "DEFAULT_MODEL",
    "DEFAULT_RUNS_DIR",
    "SWE_CODEX_PROMPT",
]


# ---------------------------------------------------------------------------
# In-process adapter app lifecycle
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class AdapterSession:
    adapter: Any  # OpenAIAdapter
    debug_handle: Any  # _dump_helpers.Debug
    port: int
    runner: Any  # aiohttp.web.AppRunner


@contextlib.asynccontextmanager
async def start_adapter_app(args, batch_dir: Path, tokenizer):
    """Bring up an in-process :class:`OpenAIAdapter` HTTP app and yield an :class:`AdapterSession`."""
    from aiohttp import web
    from tests.test_codex_agent._dump_helpers import install_dump_layer

    from slime.agent.adapters.openai import OpenAIAdapter

    batch_dir = Path(batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)

    adapter = OpenAIAdapter(
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
    site = web.TCPSite(runner, "0.0.0.0", args.port)
    await site.start()
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
# drain_and_dump_sid
# ---------------------------------------------------------------------------


async def drain_and_dump_sid(
    *,
    adapter,
    debug_handle,
    sid: str,
    inst_dir: Path,
    sample: dict[str, Any],
) -> dict[str, Any]:
    """Drain a session, write its tree + trajectory artefacts, return summary."""
    from tests.test_codex_agent._analysis import compute_tree_stats

    inst_dir = Path(inst_dir)
    inst_dir.mkdir(parents=True, exist_ok=True)
    debug_handle.sid_dump_dir[sid] = str(inst_dir)

    try:
        debug_handle.on_drain_start(sid, adapter.manager)
    except Exception as e:
        print(f"[drain] on_drain_start(sid={sid}) failed: {e}", file=sys.stderr)

    samples = await adapter.finish_session(
        sid,
        reward=0.0,
        extra_metadata={"label": sample.get("label")} if sample.get("label") is not None else None,
    )

    try:
        debug_handle.on_drain_done(sid, samples)
    except Exception as e:
        print(f"[drain] on_drain_done(sid={sid}) failed: {e}", file=sys.stderr)

    partial: dict[str, Any] = {
        "num_samples": len(samples),
        "tito_dropped_tokens": 0,
        "tito_dropped_turns": 0,
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


_DEFAULT_NODE_TGZ = Path(
    os.environ.get(
        "SWE_HOST_NODE_TARBALL",
        "/mnt/jingshenghang/software/node-v22.20.0-linux-x64.tar.xz",
    )
)
_DEFAULT_CODEX_TGZ = Path(
    os.environ.get(
        "SWE_HOST_CODEX_TARBALL",
        "/mnt/jingshenghang/code/slime_swe/artifacts/codex-0.30.0.tgz",
    )
)
DEFAULT_DATASET = Path("/mnt/jingshenghang/storage/datasets/ca/data/swe-train-1545-slime.jsonl")
DEFAULT_MODEL = "/mnt/jingshenghang/storage/checkpoints/Qwen3.6-35B-A3B"
DEFAULT_RUNS_DIR = Path("/mnt/jingshenghang/code/slime_swe/0603-trajectory-manager/runs/swe_codex")
SWE_CODEX_PROMPT = (
    "Read PROBLEM_STATEMENT.md in the current directory and resolve the issue. "
    "Edit source files only (do NOT touch tests). After editing, run the relevant "
    "tests to verify your fix passes. Do NOT modify PROBLEM_STATEMENT.md and do "
    "NOT commit. When finished, print a one-line summary and exit."
)
NODE_ENV = {
    "PATH": "/opt/node22/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "NODE_PATH": "/opt/node22/lib/node_modules",
}


async def _install_node_and_codex(sb, args, log_prefix: str) -> None:
    if not args.node_tgz.exists():
        raise FileNotFoundError(f"--node-tgz not found: {args.node_tgz}")
    if not args.codex_tgz.exists():
        raise FileNotFoundError(f"--codex-tgz not found: {args.codex_tgz}")
    await sb.write_file("/tmp/node.tar.xz", args.node_tgz)
    await sb.write_file("/tmp/codex.tgz", args.codex_tgz)
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
    # ``--prefix=/usr/local`` keeps the npm bins out of /usr (see memory:
    # e2b/e2b_sweap_image_node_and_npm_quirks).
    await sb.exec(
        "/opt/node22/bin/npm install -g --prefix=/usr/local /tmp/codex.tgz",
        env=NODE_ENV,
        timeout=600,
        check=True,
    )
    rc, out, _ = await sb.exec("codex --version", env=NODE_ENV, timeout=30, check=True)
    print(f"[{log_prefix}] codex --version: {out.strip()}")


def _build_codex_config_toml(*, host_ip: str, port: int, model: str) -> str:
    """Render the ``~/.codex/config.toml`` payload.

    ``base_url`` MUST be inline in TOML — Codex 0.30.0 only honours env vars
    for the default OpenAI provider; custom providers fall back to
    ``api.openai.com`` if ``base_url`` is missing (see memory
    ``swe-rollout/codex_version_and_config``).
    """
    # The model name written here is purely cosmetic for the SDK; the slime
    # adapter ignores ``model`` and tokenizes / serves whatever model the
    # upstream sglang has loaded.
    model_label = Path(model).name or "slime-actor"
    return (
        f'model = "{model_label}"\n'
        'model_provider = "slime"\n'
        "\n"
        "[model_providers.slime]\n"
        'name = "slime"\n'
        f'base_url = "http://{host_ip}:{port}/v1"\n'
        'env_key = "OPENAI_API_KEY"\n'
        'wire_api = "chat"\n'
    )


async def _bootstrap_agent_user(sb, workdir: str, *, host_ip: str, port: int, model: str) -> None:
    """Create the ``agent`` user, drop ``~/.codex/config.toml``, prep workdir."""
    import base64
    import shlex as _shlex

    toml_payload = _build_codex_config_toml(host_ip=host_ip, port=port, model=model)
    toml_b64 = base64.b64encode(toml_payload.encode("utf-8")).decode("ascii")
    script = (
        "set -e; "
        "id agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent; "
        f"mkdir -p /home/agent/.codex {workdir}; "
        # base64 round-trip avoids any single-quote / heredoc shell-quoting trap.
        f"echo {_shlex.quote(toml_b64)} | base64 -d > /home/agent/.codex/config.toml; "
        f"chown -R agent:agent /home/agent {workdir}; "
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
            f"sandbox cannot reach host adapter at {url} (curl rc={rc}, http={code!r}, err={err[:200]!r})"
        )


async def _run_codex(sb, args, *, sid: str, workdir: str, inst_dir: Path) -> int:
    import shlex as _shlex

    env = {
        **NODE_ENV,
        "HOME": "/home/agent",
        # Codex CLI propagates OPENAI_API_KEY into ``Authorization: Bearer``;
        # the slime adapter then resolves the sid from that header.
        "OPENAI_API_KEY": sid,
        # Disable any local key/keychain probing.
        "OPENAI_BASE_URL": f"http://{args.host_ip}:{args.port}/v1",
    }
    # ``codex exec`` is the non-interactive entrypoint shipped with the
    # Codex CLI (0.30.0). --skip-git-repo-check lets it run in plain
    # workdirs (the SWE workdir is a checked-out repo, but the check
    # is brittle against shallow clones).
    inner = f"cd {_shlex.quote(workdir)} && " f"codex exec --skip-git-repo-check {_shlex.quote(args.codex_prompt)}"
    cmd = f"runuser -u agent -- bash -c {_shlex.quote(inner)} < /dev/null"
    rc, out, err = await sb.exec(cmd, env=env, timeout=args.codex_timeout)
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
    """Run one SWE instance against the in-process adapter on ``session.port``."""
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
            await _install_node_and_codex(sb, args, log_prefix=f"{idx:04d}")
            await _bootstrap_agent_user(
                sb,
                sample["workdir"],
                host_ip=args.host_ip,
                port=args.port,
                model=args.model,
            )
            await _apply_pre_commands(sb, sample["workdir"], sample["pre_commands"])
            await sb.write_file(
                f"{sample['workdir']}/PROBLEM_STATEMENT.md",
                sample["problem_statement"] or "",
                user="agent",
            )
            await _reachability_check(sb, args.host_ip, args.port)
            rc = await _run_codex(
                sb,
                args,
                sid=sid,
                workdir=sample["workdir"],
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
    """Batch driver: load dataset, bring up adapter app, fan out workers, aggregate summary."""
    import asyncio
    import time
    import traceback

    from tests.test_codex_agent._analysis import write_summary
    from transformers import AutoTokenizer

    ensure_e2b_env()
    ts = time.strftime("%Y%m%d-%H%M%S")
    batch_dir = (Path(args.runs_dir) / ts).resolve()
    batch_dir.mkdir(parents=True)
    print(f"[swe_codex_e2e] batch_dir = {batch_dir}")

    rows = load_dataset(args.dataset, args.offset, args.limit)
    print(f"[swe_codex_e2e] loaded {len(rows)} rows (offset={args.offset}, limit={args.limit})")

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

    print(f"[swe_codex_e2e] loading tokenizer from {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    results: list[dict] = []

    async with start_adapter_app(args, batch_dir, tok) as session:
        print(f"[swe_codex_e2e] adapter app serving on {args.host_ip}:{session.port}")
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
                    print(f"[{i:04d}] unhandled worker exception: {e!r}", file=sys.stderr)
                results.append(r)
                try:
                    write_summary(batch_dir, sorted(results, key=lambda r: r.get("idx", 0)))
                except Exception:
                    pass

        await asyncio.gather(*(worker(i + args.offset, s) for i, s in enumerate(rows)))

    results.sort(key=lambda r: r.get("idx", 0))
    write_summary(batch_dir, results)
    print(f"[swe_codex_e2e] done. {len(results)} instances. summary at {batch_dir / 'summary.txt'}")
    return 0
