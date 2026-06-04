"""TDD tests for tests/test_coding_agent/_analysis.py.

Covers:
  * compute_tree_stats: walk a trajectory_tree dict and count nodes / leaves
    / forks / max_depth, plus collect per-fork detail rows.
  * write_summary: aggregate a list of per-instance result dicts into
    summary.json + summary.txt with totals over forks and TITO drops.
  * compare: cross-batch diff against a baseline runs/swe directory.

These mirror the analysis path that lived in launch_swe.py
(compute_tree_stats / write_summary) so cross-batch totals stay comparable
when we feed runs/swe_new through this module.
"""

from __future__ import annotations

import json

# ---------------------------------------------------------------------------
# compute_tree_stats
# ---------------------------------------------------------------------------


def test_compute_tree_stats_missing_session():
    """A tree dict with ``found: False`` (no session) must return
    ``{"found": False}`` and not crash."""
    from tests.test_coding_agent._analysis import compute_tree_stats

    out = compute_tree_stats({"sid": "x", "found": False})
    assert out == {"found": False}


def test_compute_tree_stats_single_leaf_chain():
    """Tree with a single linear chain (no forks): root -> system -> user ->
    assistant. n_forks=0, n_leaves=1, max_depth=3."""
    from tests.test_coding_agent._analysis import compute_tree_stats

    tree = {
        "found": True,
        "turns": 1,
        "leaves": 1,
        "nodes_total": 3,
        "root": {
            "role": None,
            "children": [
                {
                    "role": "system",
                    "children": [
                        {
                            "role": "user",
                            "children": [
                                {"role": "assistant", "children": []},
                            ],
                        }
                    ],
                }
            ],
        },
    }
    stats = compute_tree_stats(tree)
    assert stats["found"] is True
    assert stats["turns"] == 1
    assert stats["leaves"] == 1
    assert stats["nodes_total"] == 3
    assert stats["computed_leaves"] == 1
    assert stats["computed_nodes"] == 4  # root + system + user + assistant
    assert stats["max_depth"] == 3
    assert stats["n_forks"] == 0
    assert stats["forks"] == []


def test_compute_tree_stats_detects_fork_at_user_node():
    """A user node with two assistant children must register as one fork at
    that depth, with role='user' and n_children=2."""
    from tests.test_coding_agent._analysis import compute_tree_stats

    tree = {
        "found": True,
        "turns": 2,
        "leaves": 2,
        "nodes_total": 5,
        "root": {
            "role": None,
            "children": [
                {
                    "role": "system",
                    "children": [
                        {
                            "role": "user",
                            "children": [
                                {"role": "assistant", "children": []},
                                {"role": "assistant", "children": []},
                            ],
                        }
                    ],
                }
            ],
        },
    }
    stats = compute_tree_stats(tree)
    assert stats["computed_leaves"] == 2
    assert stats["n_forks"] == 1
    assert stats["forks"] == [
        {
            "depth": 2,  # user node sits at depth 2 (root=0, system=1, user=2)
            "role": "user",
            "n_children": 2,
            "child_roles": ["assistant", "assistant"],
        }
    ]


# ---------------------------------------------------------------------------
# write_summary
# ---------------------------------------------------------------------------


def test_write_summary_writes_json_and_txt(tmp_path):
    """Given a list of per-instance result dicts, write_summary must produce
    summary.json (the raw list) and summary.txt (a human-readable table with
    totals over forks, TITO drops, and snapshot recoveries)."""
    from tests.test_coding_agent._analysis import write_summary

    results = [
        {
            "idx": 0,
            "instance_id": "inst-a",
            "rc": 0,
            "elapsed_sec": 12.3,
            "tito_dropped_turns": 0,
            "tito_dropped_tokens": 0,
            "tito_snapshots_count": 0,
            "tito_snapshot_loss_tokens": 0,
            "tito_snapshot_turns": [],
            "tree": {"found": True, "turns": 3, "leaves": 1, "nodes_total": 5, "n_forks": 0, "forks": []},
        },
        {
            "idx": 1,
            "instance_id": "inst-b",
            "rc": 0,
            "elapsed_sec": 8.5,
            "tito_dropped_turns": 2,
            "tito_dropped_tokens": 1500,
            "tito_snapshots_count": 1,
            "tito_snapshot_loss_tokens": 500,
            "tito_snapshot_turns": [7],
            "tree": {
                "found": True,
                "turns": 5,
                "leaves": 3,
                "nodes_total": 9,
                "n_forks": 1,
                "forks": [
                    {"depth": 4, "role": "assistant", "n_children": 2, "child_roles": ["tool", "tool"]},
                ],
            },
        },
        {
            "idx": 2,
            "instance_id": "inst-c",
            "rc": 1,
            "error": "RuntimeError: boom",
        },
    ]
    write_summary(tmp_path, results)

    summary_json_path = tmp_path / "summary.json"
    summary_txt_path = tmp_path / "summary.txt"
    assert summary_json_path.is_file()
    assert summary_txt_path.is_file()

    raw = json.loads(summary_json_path.read_text())
    assert isinstance(raw, list)
    assert len(raw) == 3
    assert raw[1]["instance_id"] == "inst-b"

    txt = summary_txt_path.read_text()
    assert "ok=2" in txt  # 2 instances with found=True and no error
    assert "err=1" in txt
    assert "with_fork=1" in txt
    assert "with_tito_drop=1" in txt
    assert "with_snapshot=1" in txt
    assert "total_forks=1" in txt
    assert "total_tito_dropped_turns=2" in txt
    assert "total_tito_dropped_tokens=1500" in txt
    assert "total_snapshots=1" in txt
    assert "total_snapshot_saved_loss_tokens=500" in txt
    # The per-instance table should reference all 3 ids.
    assert "inst-a" in txt
    assert "inst-b" in txt
    assert "inst-c" in txt
    # Errored row should annotate the error.
    assert "ERR=RuntimeError: boom" in txt
    # Fork detail section should mention the depth/role of the fork.
    assert "fork @depth=4 role=assistant" in txt
    # Snapshot detail line lists triggered turn idx.
    assert "snapshots=1" in txt
    assert "saved_loss_tokens=500" in txt
    assert "triggered_at_turns=[7]" in txt


def test_write_summary_backwards_compatible_without_snapshot_keys(tmp_path):
    """A result dict missing tito_snapshots_count / tito_snapshot_loss_tokens
    (e.g. produced before snapshot tracking landed) must still aggregate to
    zeros without raising."""
    from tests.test_coding_agent._analysis import write_summary

    results = [
        {
            "idx": 0,
            "instance_id": "old",
            "rc": 0,
            "tito_dropped_turns": 1,
            "tito_dropped_tokens": 100,
            "tree": {"found": True, "turns": 2, "leaves": 1, "nodes_total": 3, "n_forks": 0, "forks": []},
        }
    ]
    write_summary(tmp_path, results)
    txt = (tmp_path / "summary.txt").read_text()
    assert "with_snapshot=0" in txt
    assert "total_snapshots=0" in txt
    assert "total_snapshot_saved_loss_tokens=0" in txt


# ---------------------------------------------------------------------------
# compare (cross-batch diff)
# ---------------------------------------------------------------------------


def _make_fake_batch(dir_path, results):
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "summary.json").write_text(json.dumps(results, indent=2))


def test_compare_reports_pass_when_within_threshold(tmp_path):
    """compare() must read both summary.jsons, compute per-axis ratios
    (n_forks_total, n_dropped_turns_total, etc.), and report a verdict.
    Within +/-10% absolute deviation each, the verdict is "pass".
    """
    from tests.test_coding_agent._analysis import compare

    baseline = tmp_path / "baseline"
    new = tmp_path / "new"

    def results_template(forks, drops_t, drops_k):
        return [
            {
                "idx": 0,
                "instance_id": "i",
                "rc": 0,
                "tito_dropped_turns": drops_t,
                "tito_dropped_tokens": drops_k,
                "tree": {"found": True, "turns": 3, "leaves": 1 + forks, "n_forks": forks, "forks": []},
            }
        ]

    _make_fake_batch(baseline, results_template(forks=10, drops_t=4, drops_k=1000))
    _make_fake_batch(new, results_template(forks=11, drops_t=4, drops_k=1050))  # within 10%

    report = compare(baseline, new)
    assert report["verdict"] == "pass"
    assert report["baseline_dir"] == str(baseline)
    assert report["new_dir"] == str(new)
    assert report["baseline"]["n_forks_total"] == 10
    assert report["new"]["n_forks_total"] == 11
    # rendered text output should include the headline.
    assert "PASS" in report["text"]


def test_compare_reports_fail_when_outside_threshold(tmp_path):
    """When any axis deviates >10% absolute, the verdict is "fail" and the
    text mentions which axis broke threshold."""
    from tests.test_coding_agent._analysis import compare

    baseline = tmp_path / "baseline"
    new = tmp_path / "new"
    _make_fake_batch(
        baseline,
        [
            {
                "idx": 0,
                "instance_id": "i",
                "rc": 0,
                "tito_dropped_turns": 4,
                "tito_dropped_tokens": 1000,
                "tree": {"found": True, "turns": 3, "leaves": 1, "n_forks": 10, "forks": []},
            }
        ],
    )
    _make_fake_batch(
        new,
        [
            {
                "idx": 0,
                "instance_id": "i",
                "rc": 0,
                "tito_dropped_turns": 4,
                "tito_dropped_tokens": 1000,
                "tree": {"found": True, "turns": 3, "leaves": 1, "n_forks": 3, "forks": []},
            }
        ],
    )  # -70% on n_forks
    report = compare(baseline, new)
    assert report["verdict"] == "fail"
    assert "n_forks_total" in report["text"]
    assert "FAIL" in report["text"]


def test_compare_flags_snapshot_axis_deviation(tmp_path):
    """When snapshot totals diverge >10%, the new snapshot axes must show
    up in the breached list."""
    from tests.test_coding_agent._analysis import compare

    baseline = tmp_path / "baseline"
    new = tmp_path / "new"
    _make_fake_batch(
        baseline,
        [
            {
                "idx": 0,
                "instance_id": "i",
                "rc": 0,
                "tito_snapshots_count": 10,
                "tito_snapshot_loss_tokens": 5000,
                "tree": {"found": True, "n_forks": 0, "forks": []},
            }
        ],
    )
    _make_fake_batch(
        new,
        [
            {
                "idx": 0,
                "instance_id": "i",
                "rc": 0,
                "tito_snapshots_count": 30,  # +200%
                "tito_snapshot_loss_tokens": 15000,
                "tree": {"found": True, "n_forks": 0, "forks": []},
            }
        ],
    )
    report = compare(baseline, new)
    assert report["verdict"] == "fail"
    text = report["text"]
    assert "n_snapshots_total" in text
    assert "n_snapshot_loss_tokens_total" in text
    assert report["baseline"]["n_snapshots_total"] == 10
    assert report["new"]["n_snapshots_total"] == 30


# ---------------------------------------------------------------------------
# sample_metadata_stats
# ---------------------------------------------------------------------------


def test_sample_metadata_stats_walks_trajectory_json(tmp_path):
    """sample_metadata_stats must scan each <batch>/<inst>/trajectory.json,
    split main vs snapshot Samples by ``tito_snapshot`` flag, and report
    per-instance + total counts."""
    from tests.test_coding_agent._analysis import sample_metadata_stats

    # Instance 0: 1 main with drop + 1 snapshot at turn 4 with 500 loss tokens.
    inst0 = tmp_path / "0000_alpha"
    inst0.mkdir()
    (inst0 / "trajectory.json").write_text(
        json.dumps(
            [
                {
                    "metadata": {
                        "tito_snapshot": True,
                        "tito_snapshot_at_turn": 4,
                        "tito_snapshot_loss_tokens": 500,
                    }
                },
                {
                    "metadata": {
                        "tito_dropped_turns": 2,
                        "tito_dropped_tokens": 1234,
                    }
                },
            ]
        )
    )
    # Instance 1: 1 main, no drops, no snapshots.
    inst1 = tmp_path / "0001_beta"
    inst1.mkdir()
    (inst1 / "trajectory.json").write_text(json.dumps([{"metadata": {}}]))
    # Instance 2: missing trajectory.json — should be silently skipped.
    (tmp_path / "0002_missing").mkdir()

    report = sample_metadata_stats(tmp_path)
    totals = report["totals"]
    assert totals["instances"] == 2
    assert totals["main_samples"] == 2
    assert totals["snapshot_samples"] == 1
    assert totals["main_with_drop"] == 1
    assert totals["dropped_turns"] == 2
    assert totals["dropped_tokens"] == 1234
    assert totals["snapshot_loss_tokens"] == 500

    # text view should list both instances with present trajectory.json.
    assert "0000_alpha" in report["text"]
    assert "0001_beta" in report["text"]
    assert "TOTAL instances=2" in report["text"]
