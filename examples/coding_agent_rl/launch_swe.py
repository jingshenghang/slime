"""Batch-run SWE dataset rows through a SHARED middleware + AnthropicAdapter.

Reads N rows from a slime SWE jsonl, starts ONE middleware on port 18080
(ML platform only opens that port for sandbox->host reverse), then fans out
asyncio workers each with their own E2B sandbox and unique Bearer-token sid.
The middleware's shared AnthropicAdapter + TrajectoryManager keep the trees
apart by sid; each worker registers its sid->inst_dir mapping with the
middleware before its sandbox dials in, so per-turn debug dumps (request,
response, sglang pair, openai-shape, trajectory_tree, trajectory) land
directly under ``runs/swe/<timestamp>/<idx>_<id>/`` alongside meta.json /
stdout.log instead of a separate ``mw/`` subtree.

Per-instance metadata (sid, sandbox_id, rc, tree stats) is written to
``runs/swe/<timestamp>/inst_<idx>_<id>/summary.json``; a top-level
``summary.{json,txt}`` highlights forks + TITO drops across the batch.

The middleware enforces a per-sid 100-turn cap via HTTP 429 so a runaway
agent can't burn the budget.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import aiohttp

DEFAULT_NODE_TGZ = Path(
    os.environ.get(
        "SWE_HOST_NODE_TARBALL",
        "/mnt/jingshenghang/software/node-v22.20.0-linux-x64.tar.xz",
    )
)
DEFAULT_CC_TGZ = Path(
    os.environ.get(
        "SWE_HOST_CC_TARBALL",
        "/mnt/jingshenghang/storage/claude_code/anthropic-ai-claude-code-2.1.143-local-linux-x64.tgz",
    )
)
DEFAULT_DATASET = Path("/mnt/jingshenghang/storage/datasets/ca/data/swe-train-1545-slime.jsonl")
DEFAULT_MODEL = "/mnt/jingshenghang/storage/checkpoints/Qwen3.6-35B-A3B"
DEFAULT_SANDBOX_METADATA_FILE = "/mnt/jingshenghang/code/slime_swe/0521/configs/sandbox_metadata.example.json"
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="JSONL with slime SWE rows (label/prompt/metadata.remote_env_info)",
    )
    p.add_argument("--limit", type=int, default=20, help="Process only the first N rows")
    p.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N rows before applying --limit",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Number of instances running in parallel against the single "
        "shared middleware. Each worker has its own sandbox + sid; the "
        "middleware routes by sid (Bearer token).",
    )
    p.add_argument(
        "--max-turns-per-sid",
        type=int,
        default=100,
        help="Middleware caps each sid at this many /v1/messages turns.",
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tool-parser", default="qwen3_coder")
    p.add_argument("--reasoning-parser", default="qwen3")
    p.add_argument("--sglang-url", default="http://127.0.0.1:30000")
    p.add_argument(
        "--host-ip",
        default=os.environ.get("SLIME_HEAD_HOST", "172.27.14.123"),
        help="IP that sandboxes use to dial back to this host's middleware. "
        "Default 172.27.14.123 is the dev-machine VIP; override via "
        "$SLIME_HEAD_HOST or --host-ip for other nodes.",
    )
    p.add_argument(
        "--port",
        type=int,
        default=18080,
        help="Shared middleware listen port. 18080 is the only port the ML "
        "platform opens for sandbox->host reverse.",
    )
    p.add_argument("--node-tgz", type=Path, default=DEFAULT_NODE_TGZ)
    p.add_argument("--cc-tgz", type=Path, default=DEFAULT_CC_TGZ)
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("/mnt/jingshenghang/code/slime_swe/0603-trajectory-manager/runs/swe"),
        help="Where per-batch run dirs land. Defaults to the shared "
        "0603-trajectory-manager runs root so dev-machine debug logs survive.",
    )
    p.add_argument(
        "--sandbox-timeout",
        type=int,
        default=1800,
        help="E2B sandbox total lifetime (s)",
    )
    p.add_argument(
        "--claude-timeout",
        type=int,
        default=1500,
        help="Per-instance budget for the claude-code call (s). Should be < "
        "sandbox-timeout to leave room for cleanup.",
    )
    p.add_argument(
        "--cc-prompt",
        default=SWE_CC_PROMPT,
        help="The `claude -p ...` driver prompt; the real task is in PROBLEM_STATEMENT.md",
    )
    return p.parse_args()


def ensure_e2b_env() -> None:
    # The official E2B SDK validates the API key format locally
    # (^e2b_[0-9a-f]{40}$). Internal gateways like GLM ignore the value but
    # the SDK still rejects garbage. Substitute a syntactically valid dummy
    # if the env value (or absence) fails the check.
    key = os.environ.get("E2B_API_KEY", "")
    if not re.fullmatch(r"e2b_[0-9a-fA-F]{40}", key):
        os.environ["E2B_API_KEY"] = "e2b_" + "0" * 40
    defaults = {
        "SWE_SANDBOX_METADATA_FILE": DEFAULT_SANDBOX_METADATA_FILE,
        "SWE_SANDBOX_IMAGE_METADATA_KEY": "glm-platform/image",
    }
    for k, v in defaults.items():
        os.environ.setdefault(k, v)


def normalize_sample(raw: dict[str, Any]) -> dict[str, Any]:
    """Project a swe-train jsonl row to the fields launch_swe.py needs."""
    rem = (raw.get("metadata") or {}).get("remote_env_info") or {}
    prompt_obj = raw.get("prompt") or []
    if isinstance(prompt_obj, list) and prompt_obj:
        first = prompt_obj[0]
        problem_statement = first.get("content", "") if isinstance(first, dict) else ""
    else:
        problem_statement = str(prompt_obj)
    return {
        "instance_id": rem.get("instance_id") or "unknown",
        "image": rem.get("image_url"),
        "workdir": rem.get("workdir"),
        "pre_commands": rem.get("pre_commands"),
        "problem_statement": problem_statement,
        "label": raw.get("label"),
    }


def load_dataset(path: Path, offset: int, limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if len(out) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[load] skipping line {i}: bad json: {e}", file=sys.stderr)
                continue
            out.append(normalize_sample(raw))
    return out


# ---------------------------------------------------------------------------
# Shared middleware lifecycle (one process for the whole batch)
# ---------------------------------------------------------------------------


async def wait_healthz(port: int, *, timeout: float = 60.0) -> None:
    url = f"http://127.0.0.1:{port}/healthz"
    deadline = asyncio.get_event_loop().time() + timeout
    last_err = ""
    async with aiohttp.ClientSession() as sess:
        while asyncio.get_event_loop().time() < deadline:
            try:
                async with sess.get(url, timeout=aiohttp.ClientTimeout(total=2)) as r:
                    if r.status == 200:
                        return
                    last_err = f"status={r.status}"
            except Exception as e:
                last_err = repr(e)
            await asyncio.sleep(0.5)
    raise TimeoutError(f"middleware healthz never came up on {url}: {last_err}")


async def start_shared_middleware(args: argparse.Namespace, batch_dir: Path) -> subprocess.Popen:
    """Start examples/coding_agent_rl/middleware.py as a subprocess."""
    cmd = [
        sys.executable,
        "-m",
        "examples.coding_agent_rl.middleware",
        "--port",
        str(args.port),
        "--model",
        args.model,
        "--sglang-url",
        args.sglang_url,
        "--tool-parser",
        args.tool_parser,
        "--reasoning-parser",
        args.reasoning_parser,
        "--run-dir",
        str(batch_dir),
        "--max-turns-per-sid",
        str(args.max_turns_per_sid),
    ]
    print(f"[launch_swe] starting shared middleware: {' '.join(cmd)}")
    log_fp = open(batch_dir / "middleware.log", "wb")

    repo_root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "NO_PROXY": "127.0.0.1,localhost," + os.environ.get("NO_PROXY", ""),
        "no_proxy": "127.0.0.1,localhost," + os.environ.get("no_proxy", ""),
        "PYTHONPATH": str(repo_root) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    proc = subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT, env=env)
    try:
        await wait_healthz(args.port)
        print(f"[launch_swe] middleware healthy on :{args.port}")
    except Exception:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        log_fp.close()
        tail = (batch_dir / "middleware.log").read_text(errors="replace")[-2000:]
        print(
            f"[launch_swe] middleware failed to come up. tail:\n{tail}",
            file=sys.stderr,
        )
        raise
    return proc


async def drain_sid_trajectory(
    port: int,
    sid: str,
    *,
    sample_index: int,
    prompt: str,
    reward: float,
    label: str | None,
) -> dict[str, Any] | None:
    """POST /get_trajectory for ONE sid; middleware writes <sid>/trajectory.*"""
    body: dict[str, Any] = {
        "sid": sid,
        "index": sample_index,
        "reward": reward,
        "prompt": prompt,
    }
    if label is not None:
        body["label"] = label
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.post(
                f"http://127.0.0.1:{port}/get_trajectory",
                json=body,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as r:
                if r.status != 200:
                    txt = await r.text()
                    print(f"[sid={sid}] /get_trajectory -> {r.status}: {txt[:200]}")
                    return None
                payload = await r.json()
                ns = (payload or {}).get("num_samples")
                print(f"[sid={sid}] /get_trajectory -> 200 num_samples={ns}")
                return payload
    except Exception as e:
        print(f"[sid={sid}] /get_trajectory failed: {e}", file=sys.stderr)
        return None


async def register_sid_with_middleware(port: int, sid: str, dump_dir: Path) -> None:
    """Tell the shared middleware to write this sid's debug dumps into ``dump_dir``.

    Must be called BEFORE cc inside the sandbox issues its first /v1/messages
    request, otherwise early-turn dumps fall back to ``run_dir/<sid>/``.
    Failure is non-fatal: a missing registration just means dumps for this
    sid land in the fallback location.
    """
    url = f"http://127.0.0.1:{port}/register_sid"
    body = {"sid": sid, "dump_dir": str(dump_dir.resolve())}
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.post(
                url,
                json=body,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as r:
                if r.status != 200:
                    txt = await r.text()
                    print(
                        f"[sid={sid}] /register_sid -> {r.status}: {txt[:200]}",
                        file=sys.stderr,
                    )
    except Exception as e:
        print(f"[sid={sid}] /register_sid failed: {e}", file=sys.stderr)


async def shutdown_middleware(mw: subprocess.Popen) -> None:
    mw.send_signal(signal.SIGTERM)
    try:
        mw.wait(timeout=10)
    except subprocess.TimeoutExpired:
        print(
            "[launch_swe] middleware did not exit after SIGTERM; killing",
            file=sys.stderr,
        )
        mw.kill()


# ---------------------------------------------------------------------------
# Sandbox prep + run
# ---------------------------------------------------------------------------


async def install_node_and_cc(sb, args: argparse.Namespace, log_prefix: str) -> None:
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


async def bootstrap_agent_user(sb, workdir: str) -> None:
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


async def apply_pre_commands(sb, workdir: str, pre: Any) -> None:
    if not pre:
        return
    if isinstance(pre, str):
        body = pre.replace("\\n", "\n")
    else:
        body = "\n".join(c for c in pre if c)
    await sb.write_file("/workspace/__cagent_pre__.sh", "set -e\n" + body, user="agent")
    await sb.exec(
        f"chmod 755 /workspace/__cagent_pre__.sh && "
        f"cd {shlex.quote(workdir)} && bash /workspace/__cagent_pre__.sh",
        user="agent",
        check=False,
        timeout=600,
    )


async def write_problem_statement(sb, workdir: str, text: str) -> None:
    await sb.write_file(f"{workdir}/PROBLEM_STATEMENT.md", text or "", user="agent")


async def reachability_check(sb, host_ip: str, port: int) -> None:
    url = f"http://{host_ip}:{port}/healthz"
    rc, out, err = await sb.exec(
        f"curl -sS --max-time 5 -o /dev/null -w '%{{http_code}}' {shlex.quote(url)}",
        env=NODE_ENV,
        timeout=15,
    )
    code = out.strip()
    if rc != 0 or code != "200":
        raise RuntimeError(
            f"sandbox cannot reach host middleware at {url} (curl rc={rc}, " f"http={code!r}, err={err[:200]!r})"
        )


async def run_claude(
    sb,
    args: argparse.Namespace,
    *,
    sid: str,
    workdir: str,
    model: str,
    inst_dir: Path,
) -> int:
    env = {
        **NODE_ENV,
        "HOME": "/home/agent",
        "ANTHROPIC_BASE_URL": f"http://{args.host_ip}:{args.port}",
        "ANTHROPIC_AUTH_TOKEN": sid,
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_API_KEY": "dummy",
    }
    inner = (
        f"cd {shlex.quote(workdir)} && "
        f"claude -p {shlex.quote(args.cc_prompt)} "
        f"--permission-mode bypassPermissions"
    )
    cmd = f"runuser -u agent -- bash -c {shlex.quote(inner)} " f"< /dev/null"
    rc, out, err = await sb.exec(cmd, env=env, timeout=args.claude_timeout)
    (inst_dir / "stdout.log").write_text(out, encoding="utf-8")
    (inst_dir / "stderr.log").write_text(err, encoding="utf-8")
    return rc


# ---------------------------------------------------------------------------
# Per-instance worker
# ---------------------------------------------------------------------------


async def run_one_instance(
    args: argparse.Namespace,
    batch_dir: Path,
    idx: int,
    sample: dict[str, Any],
) -> dict[str, Any]:
    """One sample worker. Boots its own sandbox, runs cc against the shared
    middleware on args.port, then asks the middleware to drain its sid."""
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
        "sglang_url": args.sglang_url,
        "port": args.port,
        "max_turns_per_sid": args.max_turns_per_sid,
    }
    (inst_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    (inst_dir / "PROBLEM_STATEMENT.md").write_text(sample["problem_statement"], encoding="utf-8")

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
            await install_node_and_cc(sb, args, log_prefix=f"{idx:04d}")
            await bootstrap_agent_user(sb, sample["workdir"])
            await apply_pre_commands(sb, sample["workdir"], sample["pre_commands"])
            await write_problem_statement(sb, sample["workdir"], sample["problem_statement"])
            await reachability_check(sb, args.host_ip, args.port)
            await register_sid_with_middleware(args.port, sid, inst_dir)
            rc = await run_claude(
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
        print(
            f"[{idx:04d}] FAILED during sandbox phase: {e!r}",
            file=sys.stderr,
        )
    finally:
        # Drain this sid from the shared middleware. Other workers can still
        # be running their own sids against the same middleware.
        await drain_sid_trajectory(
            args.port,
            sid,
            sample_index=idx,
            prompt=sample["problem_statement"][:8000],
            reward=0.0,
            label=sample["label"],
        )
        summary["elapsed_sec"] = round(time.time() - t0, 1)

    # The middleware wrote trajectory_tree.json + trajectory.json directly
    # into inst_dir (sid registered via /register_sid before cc started).
    sid_dump_dir = inst_dir
    tree_json = sid_dump_dir / "trajectory_tree.json"
    if tree_json.exists():
        try:
            tree = json.loads(tree_json.read_text())
            summary["tree"] = compute_tree_stats(tree)
        except Exception as e:
            summary["tree"] = {"error": f"parse: {e}"}
    else:
        summary["tree"] = {"error": "no_trajectory_tree_json"}

    traj_json = sid_dump_dir / "trajectory.json"
    if traj_json.exists():
        try:
            samples = json.loads(traj_json.read_text())
            total_dropped = 0
            dropped_turns = 0
            for s in samples or []:
                md = s.get("metadata") or {}
                total_dropped += int(md.get("tito_dropped_tokens", 0) or 0)
                dropped_turns += int(md.get("tito_dropped_turns", 0) or 0)
            summary["tito_dropped_tokens"] = total_dropped
            summary["tito_dropped_turns"] = dropped_turns
            summary["num_samples"] = len(samples or [])
        except Exception as e:
            summary["trajectory_parse_error"] = str(e)

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


def compute_tree_stats(tree: dict) -> dict:
    """Count nodes, leaves, max depth, and forks."""
    if not tree.get("found"):
        return {"found": False}
    root = tree.get("root") or {}

    n_nodes = 0
    n_leaves = 0
    n_forks = 0
    forks_detail: list[dict] = []
    max_depth = 0

    def walk(node: dict, depth: int) -> None:
        nonlocal n_nodes, n_leaves, n_forks, max_depth
        n_nodes += 1
        max_depth = max(max_depth, depth)
        kids = node.get("children") or []
        if not kids:
            n_leaves += 1
            return
        if len(kids) > 1:
            n_forks += 1
            forks_detail.append(
                {
                    "depth": depth,
                    "role": node.get("role"),
                    "n_children": len(kids),
                    "child_roles": [k.get("role") for k in kids],
                }
            )
        for k in kids:
            walk(k, depth + 1)

    walk(root, 0)
    return {
        "found": True,
        "turns": tree.get("turns"),
        "leaves": tree.get("leaves"),
        "nodes_total": tree.get("nodes_total"),
        "computed_nodes": n_nodes,
        "computed_leaves": n_leaves,
        "max_depth": max_depth,
        "n_forks": n_forks,
        "forks": forks_detail,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def amain() -> int:
    args = parse_args()
    ensure_e2b_env()
    ts = time.strftime("%Y%m%d-%H%M%S")
    batch_dir = (args.runs_dir / ts).resolve()
    batch_dir.mkdir(parents=True)
    print(f"[launch_swe] batch_dir = {batch_dir}")

    rows = load_dataset(args.dataset, args.offset, args.limit)
    print(f"[launch_swe] loaded {len(rows)} rows (offset={args.offset}, limit={args.limit})")

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
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    mw = await start_shared_middleware(args, batch_dir)
    sem = asyncio.Semaphore(args.concurrency)
    results: list[dict] = []

    async def worker(i: int, sample: dict[str, Any]) -> None:
        async with sem:
            try:
                r = await run_one_instance(args, batch_dir, i, sample)
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
            # Incremental on-disk summary so progress is inspectable mid-run.
            try:
                results_sorted = sorted(results, key=lambda r: r.get("idx", 0))
                write_summary(batch_dir, results_sorted)
            except Exception:
                pass

    try:
        await asyncio.gather(*(worker(i + args.offset, s) for i, s in enumerate(rows)))
    finally:
        await shutdown_middleware(mw)

    results.sort(key=lambda r: r.get("idx", 0))
    write_summary(batch_dir, results)
    print(f"[launch_swe] done. {len(results)} instances. " f"summary at {batch_dir / 'summary.txt'}")
    return 0


def write_summary(batch_dir: Path, results: list[dict]) -> None:
    (batch_dir / "summary.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    lines: list[str] = []
    lines.append(f"# launch_swe summary  ({len(results)} instances)\n")

    n_ok = sum(1 for r in results if r.get("error") is None and (r.get("tree") or {}).get("found"))
    n_err = sum(1 for r in results if r.get("error"))
    n_forks_total = sum((r.get("tree") or {}).get("n_forks", 0) for r in results)
    n_dropped_turns_total = sum(int(r.get("tito_dropped_turns") or 0) for r in results)
    n_dropped_tokens_total = sum(int(r.get("tito_dropped_tokens") or 0) for r in results)
    n_with_fork = sum(1 for r in results if (r.get("tree") or {}).get("n_forks", 0) > 0)
    n_with_drop = sum(1 for r in results if int(r.get("tito_dropped_turns") or 0) > 0)
    lines.append(f"ok={n_ok}  err={n_err}  with_fork={n_with_fork}  with_tito_drop={n_with_drop}")
    lines.append(
        f"total_forks={n_forks_total}  "
        f"total_tito_dropped_turns={n_dropped_turns_total}  "
        f"total_tito_dropped_tokens={n_dropped_tokens_total}\n"
    )

    lines.append(
        f"{'idx':>4}  {'turns':>5}  {'leaves':>6}  {'nodes':>5}  "
        f"{'forks':>5}  {'tito_t':>6}  {'tito_k':>6}  {'elapsed':>7}  rc  instance_id"
    )
    lines.append("-" * 110)
    for r in results:
        t = r.get("tree") or {}
        lines.append(
            f"{r.get('idx', 0):>4}  "
            f"{str(t.get('turns', '-')):>5}  "
            f"{str(t.get('leaves', '-')):>6}  "
            f"{str(t.get('nodes_total', '-')):>5}  "
            f"{str(t.get('n_forks', '-')):>5}  "
            f"{str(r.get('tito_dropped_turns', '-')):>6}  "
            f"{str(r.get('tito_dropped_tokens', '-')):>6}  "
            f"{str(r.get('elapsed_sec', '-')):>7}  "
            f"{str(r.get('rc', '-')):>2}  "
            f"{r.get('instance_id', '?')}" + (f"  ERR={r.get('error')}" if r.get("error") else "")
        )

    lines.append("\n## fork / drop detail\n")
    has_detail = False
    for r in results:
        t = r.get("tree") or {}
        forks = t.get("forks") or []
        drops = int(r.get("tito_dropped_turns") or 0)
        if not forks and not drops:
            continue
        has_detail = True
        lines.append(
            f"[{r['idx']:04d}] {r['instance_id']}: "
            f"forks={len(forks)} dropped_turns={drops} "
            f"dropped_tokens={r.get('tito_dropped_tokens', 0)}"
        )
        for fk in forks:
            lines.append(
                f"   fork @depth={fk['depth']} role={fk['role']} "
                f"n_children={fk['n_children']} child_roles={fk['child_roles']}"
            )
    if not has_detail:
        lines.append("(no forks or TITO drops across the batch)")

    (batch_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    try:
        rc = asyncio.run(amain())
    except KeyboardInterrupt:
        rc = 130
    sys.exit(rc)


if __name__ == "__main__":
    main()
