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
      * with_fork / with_tito_drop / with_snapshot instance counts
      * total_forks / total_tito_dropped_turns / total_tito_dropped_tokens
      * total_snapshots / total_snapshot_saved_loss_tokens

    Snapshot fields are sourced from per-instance partials populated by
    ``drain_and_dump_sid`` (``tito_snapshots_count``,
    ``tito_snapshot_loss_tokens``, ``tito_snapshot_turns``). Missing keys
    are treated as zero so summaries from before snapshot tracking still
    render.
    """
    batch_dir = Path(batch_dir)
    (batch_dir / "summary.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    totals = _aggregate_totals(results)

    lines: list[str] = []
    lines.append(f"# coding_agent_swe_e2e summary  ({len(results)} instances)\n")
    lines.append(
        f"ok={totals['n_ok']}  err={totals['n_err']}  "
        f"with_fork={totals['n_with_fork']}  with_tito_drop={totals['n_with_drop']}  "
        f"with_snapshot={totals['n_with_snapshot']}"
    )
    lines.append(
        f"total_forks={totals['n_forks_total']}  "
        f"total_tito_dropped_turns={totals['n_dropped_turns_total']}  "
        f"total_tito_dropped_tokens={totals['n_dropped_tokens_total']}"
    )
    lines.append(
        f"total_snapshots={totals['n_snapshots_total']}  "
        f"total_snapshot_saved_loss_tokens={totals['n_snapshot_loss_tokens_total']}\n"
    )

    lines.append(
        f"{'idx':>4}  {'turns':>5}  {'leaves':>6}  {'nodes':>5}  "
        f"{'forks':>5}  {'tito_t':>6}  {'tito_k':>6}  "
        f"{'snap_n':>6}  {'snap_k':>6}  {'rec%':>5}  "
        f"{'elapsed':>7}  rc  instance_id"
    )
    lines.append("-" * 130)
    for r in results:
        t = r.get("tree") or {}
        snap_n = int(r.get("tito_snapshots_count") or 0)
        snap_k = int(r.get("tito_snapshot_loss_tokens") or 0)
        drop_k = int(r.get("tito_dropped_tokens") or 0)
        denom = snap_k + drop_k
        rec_pct = f"{(snap_k * 100.0 / denom):.0f}" if denom > 0 else "-"
        lines.append(
            f"{r.get('idx', 0):>4}  "
            f"{str(t.get('turns', '-')):>5}  "
            f"{str(t.get('leaves', '-')):>6}  "
            f"{str(t.get('nodes_total', '-')):>5}  "
            f"{str(t.get('n_forks', '-')):>5}  "
            f"{str(r.get('tito_dropped_turns', '-')):>6}  "
            f"{str(drop_k):>6}  "
            f"{str(snap_n):>6}  "
            f"{str(snap_k):>6}  "
            f"{rec_pct:>5}  "
            f"{str(r.get('elapsed_sec', '-')):>7}  "
            f"{str(r.get('rc', '-')):>2}  "
            f"{r.get('instance_id', '?')}" + (f"  ERR={r.get('error')}" if r.get("error") else "")
        )

    lines.append("\n## fork / drop / snapshot detail\n")
    has_detail = False
    for r in results:
        t = r.get("tree") or {}
        forks = t.get("forks") or []
        drops = int(r.get("tito_dropped_turns") or 0)
        snaps = int(r.get("tito_snapshots_count") or 0)
        if not forks and not drops and not snaps:
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
        if snaps:
            turns_list = r.get("tito_snapshot_turns") or []
            lines.append(
                f"   snapshots={snaps} "
                f"saved_loss_tokens={r.get('tito_snapshot_loss_tokens', 0)}  "
                f"triggered_at_turns={turns_list}"
            )
    if not has_detail:
        lines.append("(no forks, TITO drops, or snapshots across the batch)")

    (batch_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _aggregate_totals(results: list[dict]) -> dict[str, int]:
    """Shared aggregation used by both ``write_summary`` and ``compare``.

    Missing snapshot keys (older summaries written before snapshot tracking)
    are treated as zero so the schema stays backwards-compatible.
    """
    n_ok = sum(1 for r in results if r.get("error") is None and (r.get("tree") or {}).get("found"))
    n_err = sum(1 for r in results if r.get("error"))
    n_forks_total = sum(int((r.get("tree") or {}).get("n_forks", 0)) for r in results)
    n_dropped_turns_total = sum(int(r.get("tito_dropped_turns") or 0) for r in results)
    n_dropped_tokens_total = sum(int(r.get("tito_dropped_tokens") or 0) for r in results)
    n_snapshots_total = sum(int(r.get("tito_snapshots_count") or 0) for r in results)
    n_snapshot_loss_tokens_total = sum(int(r.get("tito_snapshot_loss_tokens") or 0) for r in results)
    n_with_fork = sum(1 for r in results if int((r.get("tree") or {}).get("n_forks", 0)) > 0)
    n_with_drop = sum(1 for r in results if int(r.get("tito_dropped_turns") or 0) > 0)
    n_with_snapshot = sum(1 for r in results if int(r.get("tito_snapshots_count") or 0) > 0)
    return {
        "n_ok": n_ok,
        "n_err": n_err,
        "n_forks_total": n_forks_total,
        "n_dropped_turns_total": n_dropped_turns_total,
        "n_dropped_tokens_total": n_dropped_tokens_total,
        "n_snapshots_total": n_snapshots_total,
        "n_snapshot_loss_tokens_total": n_snapshot_loss_tokens_total,
        "n_with_fork": n_with_fork,
        "n_with_drop": n_with_drop,
        "n_with_snapshot": n_with_snapshot,
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
    "n_snapshots_total",
    "n_snapshot_loss_tokens_total",
    "n_with_snapshot",
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


def sample_metadata_stats(batch_dir: Path | str) -> dict[str, Any]:
    """Walk ``<batch_dir>/*/trajectory.json`` and report per-sample TITO
    metadata distribution, independent of summary.json aggregation.

    For each per-instance ``trajectory.json`` (a list of sample dicts from
    :func:`Sample.to_dict`), counts:
      * number of main-leaf Samples with ``tito_dropped_*`` set
      * number of snapshot Samples (``tito_snapshot == True``)
      * sum of dropped tokens / snapshot loss tokens

    Useful for sanity-checking ``write_summary`` aggregation by re-deriving
    the same numbers from the canonical per-sample metadata on disk.

    Returns a dict with ``per_instance`` (list of {idx, instance_id, ...})
    and ``totals``.
    """
    batch_dir = Path(batch_dir)
    per_instance: list[dict[str, Any]] = []
    totals: dict[str, int] = {
        "instances": 0,
        "main_samples": 0,
        "snapshot_samples": 0,
        "main_with_drop": 0,
        "dropped_tokens": 0,
        "dropped_turns": 0,
        "snapshot_loss_tokens": 0,
    }
    inst_dirs = sorted(p for p in batch_dir.iterdir() if p.is_dir())
    for inst_dir in inst_dirs:
        traj_path = inst_dir / "trajectory.json"
        if not traj_path.is_file():
            continue
        try:
            samples = json.loads(traj_path.read_text())
        except Exception as e:
            per_instance.append({"inst_dir": inst_dir.name, "error": f"parse: {e}"})
            continue
        if not isinstance(samples, list):
            continue
        totals["instances"] += 1
        rec: dict[str, Any] = {
            "inst_dir": inst_dir.name,
            "main_samples": 0,
            "snapshot_samples": 0,
            "main_with_drop": 0,
            "dropped_tokens": 0,
            "dropped_turns": 0,
            "snapshot_loss_tokens": 0,
            "snapshot_at_turns": [],
        }
        for s in samples:
            md = s.get("metadata") if isinstance(s, dict) else None
            md = md or {}
            if md.get("tito_snapshot"):
                rec["snapshot_samples"] += 1
                k = int(md.get("tito_snapshot_loss_tokens", 0) or 0)
                rec["snapshot_loss_tokens"] += k
                at_turn = md.get("tito_snapshot_at_turn")
                if at_turn is not None:
                    rec["snapshot_at_turns"].append(int(at_turn))
            else:
                rec["main_samples"] += 1
                drop_t = int(md.get("tito_dropped_turns", 0) or 0)
                drop_k = int(md.get("tito_dropped_tokens", 0) or 0)
                if drop_t > 0 or drop_k > 0:
                    rec["main_with_drop"] += 1
                rec["dropped_turns"] += drop_t
                rec["dropped_tokens"] += drop_k
        for k in (
            "main_samples",
            "snapshot_samples",
            "main_with_drop",
            "dropped_tokens",
            "dropped_turns",
            "snapshot_loss_tokens",
        ):
            totals[k] += rec[k]
        per_instance.append(rec)

    lines = [
        f"sample-metadata-stats batch={batch_dir}",
        "",
        f"{'inst_dir':<60} {'main':>5} {'snap':>5} {'drop_t':>6} {'drop_k':>8} {'snap_k':>8}",
        "-" * 100,
    ]
    for rec in per_instance:
        if "error" in rec:
            lines.append(f"{rec['inst_dir']:<60}  ERROR {rec['error']}")
            continue
        lines.append(
            f"{rec['inst_dir']:<60} "
            f"{rec['main_samples']:>5} "
            f"{rec['snapshot_samples']:>5} "
            f"{rec['dropped_turns']:>6} "
            f"{rec['dropped_tokens']:>8} "
            f"{rec['snapshot_loss_tokens']:>8}"
        )
    lines.append("")
    lines.append(
        f"TOTAL instances={totals['instances']} "
        f"main={totals['main_samples']} snap={totals['snapshot_samples']} "
        f"main_with_drop={totals['main_with_drop']} "
        f"drop_t={totals['dropped_turns']} drop_k={totals['dropped_tokens']} "
        f"snap_k={totals['snapshot_loss_tokens']}"
    )
    return {
        "batch_dir": str(batch_dir),
        "per_instance": per_instance,
        "totals": totals,
        "text": "\n".join(lines),
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

    smd_parser = sub.add_parser(
        "sample-metadata-stats",
        help="Walk <batch>/*/trajectory.json and print per-sample TITO metadata distribution.",
    )
    smd_parser.add_argument("batch", type=Path)

    args = parser.parse_args()
    if args.cmd == "compare":
        report = compare(args.baseline, args.new)
        print(report["text"])
        return 0 if report["verdict"] == "pass" else 1
    if args.cmd == "sample-metadata-stats":
        report = sample_metadata_stats(args.batch)
        print(report["text"])
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(_cli())


__all__ = [
    "compute_tree_stats",
    "write_summary",
    "compare",
    "sample_metadata_stats",
]
