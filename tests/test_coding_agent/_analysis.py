"""Analysis helpers for the coding-agent e2e test batches.

* ``compute_tree_stats(tree)``: walk a per-instance ``trajectory_tree.json``
  dict and count nodes / leaves / forks / max depth, plus collect a
  per-fork detail row (depth, role, child roles).
* ``write_summary(batch_dir, results)``: aggregate a list of per-instance
  result dicts into ``summary.json`` (raw list) and ``summary.txt``
  (human-readable totals + per-instance table + fork/drop detail section).
* ``compare(baseline_dir, new_dir)``: read two batches' summary.json files
  and report whether fork / TITO drop totals agree within ±10% absolute.

Imported by ``test_coding_agent_swe_e2e.py``; also runnable as a CLI
``python -m tests.test_coding_agent._analysis compare --baseline ... --new ...``
for cross-batch verification.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Threshold used by ``compare`` when ruling a metric pass/fail. A baseline
# value of N and a new value of M agree iff |M-N| <= max(1, ceil(N * 0.10)).
COMPARE_TOLERANCE = 0.10


# ---------------------------------------------------------------------------
# compute_tree_stats
# ---------------------------------------------------------------------------


def compute_tree_stats(tree: dict) -> dict:
    """Walk a trajectory tree dict (as produced by ``dump_tree_json``) and
    return counts plus per-fork detail.

    Returns ``{"found": False}`` if the input has ``found`` falsy.
    """
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
        if depth > max_depth:
            max_depth = depth
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
# write_summary
# ---------------------------------------------------------------------------


def write_summary(batch_dir: Path, results: list[dict]) -> None:
    """Persist the per-instance ``results`` list to ``summary.json`` and a
    human-readable ``summary.txt`` table next to it.

    Computed top-line metrics (used by ``compare``):
      * ok / err counts
      * with_fork / with_tito_drop instance counts
      * total_forks / total_tito_dropped_turns / total_tito_dropped_tokens
    """
    batch_dir = Path(batch_dir)
    (batch_dir / "summary.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    totals = _aggregate_totals(results)

    lines: list[str] = []
    lines.append(f"# coding_agent_swe_e2e summary  ({len(results)} instances)\n")
    lines.append(
        f"ok={totals['n_ok']}  err={totals['n_err']}  "
        f"with_fork={totals['n_with_fork']}  with_tito_drop={totals['n_with_drop']}"
    )
    lines.append(
        f"total_forks={totals['n_forks_total']}  "
        f"total_tito_dropped_turns={totals['n_dropped_turns_total']}  "
        f"total_tito_dropped_tokens={totals['n_dropped_tokens_total']}\n"
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
            f"[{r.get('idx', 0):04d}] {r.get('instance_id', '?')}: "
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


def _aggregate_totals(results: list[dict]) -> dict[str, int]:
    """Shared aggregation used by both ``write_summary`` and ``compare``."""
    n_ok = sum(1 for r in results if r.get("error") is None and (r.get("tree") or {}).get("found"))
    n_err = sum(1 for r in results if r.get("error"))
    n_forks_total = sum(int((r.get("tree") or {}).get("n_forks", 0)) for r in results)
    n_dropped_turns_total = sum(int(r.get("tito_dropped_turns") or 0) for r in results)
    n_dropped_tokens_total = sum(int(r.get("tito_dropped_tokens") or 0) for r in results)
    n_with_fork = sum(1 for r in results if int((r.get("tree") or {}).get("n_forks", 0)) > 0)
    n_with_drop = sum(1 for r in results if int(r.get("tito_dropped_turns") or 0) > 0)
    return {
        "n_ok": n_ok,
        "n_err": n_err,
        "n_forks_total": n_forks_total,
        "n_dropped_turns_total": n_dropped_turns_total,
        "n_dropped_tokens_total": n_dropped_tokens_total,
        "n_with_fork": n_with_fork,
        "n_with_drop": n_with_drop,
    }


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


COMPARE_AXES = (
    "n_forks_total",
    "n_dropped_turns_total",
    "n_dropped_tokens_total",
    "n_with_fork",
    "n_with_drop",
)


def compare(baseline_dir: Path | str, new_dir: Path | str) -> dict[str, Any]:
    """Compare two batches' ``summary.json`` totals and report per-axis
    deviation. Tolerance: max(1, ceil(baseline * 10%)) on each axis.

    Returns a dict with:
      * ``baseline``  / ``new``: aggregated totals dicts
      * ``deltas``: per-axis absolute deviation
      * ``per_axis_pass``: bool per axis
      * ``verdict``: "pass" iff every axis within tolerance, else "fail"
      * ``text``: rendered human-readable report (used by CLI)
    """
    baseline_dir = Path(baseline_dir)
    new_dir = Path(new_dir)
    baseline_summary = json.loads((baseline_dir / "summary.json").read_text())
    new_summary = json.loads((new_dir / "summary.json").read_text())
    baseline_totals = _aggregate_totals(baseline_summary)
    new_totals = _aggregate_totals(new_summary)

    deltas: dict[str, int] = {}
    per_axis_pass: dict[str, bool] = {}
    breached: list[str] = []
    for axis in COMPARE_AXES:
        b = baseline_totals[axis]
        n = new_totals[axis]
        deltas[axis] = n - b
        tol = max(1, int(round(abs(b) * COMPARE_TOLERANCE + 0.4999999)))
        ok = abs(n - b) <= tol
        per_axis_pass[axis] = ok
        if not ok:
            breached.append(f"{axis}: baseline={b} new={n} (delta={n-b}, tol=±{tol})")
    verdict = "pass" if not breached else "fail"
    headline = "PASS" if verdict == "pass" else "FAIL"

    lines = [
        f"[{headline}] compare baseline={baseline_dir}  new={new_dir}",
        "",
        f"{'axis':<28} {'baseline':>10} {'new':>10} {'delta':>10}  ok",
        "-" * 70,
    ]
    for axis in COMPARE_AXES:
        ok = per_axis_pass[axis]
        lines.append(
            f"{axis:<28} {baseline_totals[axis]:>10} {new_totals[axis]:>10} "
            f"{deltas[axis]:>10}  {'YES' if ok else 'NO'}"
        )
    if breached:
        lines.append("")
        lines.append("breached axes:")
        lines.extend(f"  - {b}" for b in breached)
    text = "\n".join(lines)

    return {
        "verdict": verdict,
        "baseline_dir": str(baseline_dir),
        "new_dir": str(new_dir),
        "baseline": baseline_totals,
        "new": new_totals,
        "deltas": deltas,
        "per_axis_pass": per_axis_pass,
        "text": text,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.test_coding_agent._analysis")
    sub = parser.add_subparsers(dest="cmd", required=True)

    cmp_parser = sub.add_parser("compare", help="compare two batches' summary.json")
    cmp_parser.add_argument("--baseline", type=Path, required=True)
    cmp_parser.add_argument("--new", type=Path, required=True)

    args = parser.parse_args()
    if args.cmd == "compare":
        report = compare(args.baseline, args.new)
        print(report["text"])
        return 0 if report["verdict"] == "pass" else 1
    return 2


if __name__ == "__main__":
    sys.exit(_cli())


__all__ = [
    "compute_tree_stats",
    "write_summary",
    "compare",
]
