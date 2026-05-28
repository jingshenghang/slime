"""Fully-async wire-up for coding-agent-rl.

Both ``--rollout-function-path`` and ``--custom-generate-function-path``
resolve here; the latter re-exports :func:`generate` from
:mod:`.generate`.

Subclasses upstream :class:`AsyncRolloutWorker` to handle
``list[list[Sample]]`` groups (produced by
:func:`slime.agent.trajectory.fan_out_sample_segments`): upstream's
ABORTED check and ``_key`` sort treat a ``list`` element as a Sample
and silently mis-route / crash.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import threading
import time

from slime.rollout.fully_async_rollout import AsyncRolloutWorker
from slime.utils.async_utils import run
from slime.utils.types import Sample

from .generate import generate  # noqa: F401  re-export

__all__ = [
    "FanOutAsyncRolloutWorker",
    "generate",
    "generate_rollout_fully_async",
]

logger = logging.getLogger("slime.rollout.fully_async")


class FanOutAsyncRolloutWorker(AsyncRolloutWorker):
    """AsyncRolloutWorker that handles fan-out groups (``list[list[Sample]]``).
    Upstream's ABORTED check + requeue assumes a flat ``list[Sample]``; on
    fan-out it both misses nested aborts and re-adds a misshapen group that
    crashes the next ``generate_and_rm_group`` pickup."""

    def _make_done_cb(self, gid: int):
        def _cb(done_task: asyncio.Task) -> None:
            try:
                result = done_task.result()
            except Exception:  # noqa: BLE001
                logger.exception("fully-async: process task raised")
                return
            if not isinstance(result, list):
                logger.warning(
                    "fully-async: generate_and_rm_group returned %r, expected list[Sample]; dropping",
                    type(result).__name__,
                )
                return

            def _has_aborted(node):
                if isinstance(node, list):
                    return any(_has_aborted(x) for x in node)
                return getattr(node, "status", None) == Sample.Status.ABORTED

            if _has_aborted(result):
                retry_group = _flatten_for_retry(result)
                if retry_group:
                    try:
                        self.data_buffer.add_samples([retry_group])
                    except Exception:  # noqa: BLE001
                        logger.exception("fully-async: failed to requeue aborted group")
                return
            self.output_queue.put((gid, result))

        return _cb


def _flatten_for_retry(result: list) -> list[Sample]:
    """Recover the original ``list[Sample]`` group from a fan-out result so it
    can be re-fed through ``generate_and_rm_group``. The fan-out lists share
    the original prompt across all inner samples, so picking the first sample
    of each inner list reconstructs the K-sample group; we then reset trajectory
    state so the next pickup starts a fresh agent run."""
    group: list[Sample] = []
    for item in result:
        inner = item if isinstance(item, list) else [item]
        if not inner:
            continue
        sample = inner[0]
        if not isinstance(sample, Sample):
            continue
        sample.status = Sample.Status.PENDING
        sample.session_id = None
        sample.tokens = []
        sample.response = ""
        sample.response_length = 0
        sample.loss_mask = []
        sample.reward = 0.0
        if isinstance(sample.metadata, dict):
            sample.metadata.pop("abort_reason", None)
        group.append(sample)
    return group


# Independent of upstream's _global_worker so the two atexit hooks don't
# fight over the same handle.
_global_worker: FanOutAsyncRolloutWorker | None = None
_worker_lock = threading.Lock()


def _get_global_worker(args, data_buffer) -> FanOutAsyncRolloutWorker:
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            logger.info("starting fully-async rollout worker (fan-out subclass)")
            num_engines = max(1, args.rollout_num_gpus // args.rollout_num_gpus_per_engine)
            _global_worker = FanOutAsyncRolloutWorker(
                args, data_buffer, concurrency=args.sglang_server_concurrency * num_engines
            )
            _global_worker.start()
        return _global_worker


def _stop_global_worker() -> None:
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


atexit.register(_stop_global_worker)


async def _generate_rollout_async(args, rollout_id: int, data_buffer) -> list[list[Sample]]:
    assert args.rollout_global_dataset
    worker = _get_global_worker(args, data_buffer)

    target = args.rollout_batch_size
    logger.info(
        "fully-async rollout %d: target=%d queue_warm=%d",
        rollout_id,
        target,
        worker.queue_size(),
    )

    collected: dict[int, list[Sample]] = {}
    started = time.time()
    last_log = started
    LOG_EVERY = 30.0

    while len(collected) < target:
        drained = 0
        for gid, group in worker.get_completed_groups():
            collected[gid] = group
            drained += 1
        if not drained:
            await asyncio.sleep(0.05)
        now = time.time()
        if now - last_log > LOG_EVERY:
            logger.info(
                "fully-async rollout %d: collected %d/%d, queue=%d, elapsed=%.1fs",
                rollout_id,
                len(collected),
                target,
                worker.queue_size(),
                now - started,
            )
            last_log = now

    # Recurse into nested groups: getattr(list, "index") returns the bound
    # method, passes upstream's `is not None`, then crashes int().
    def _key(group) -> int:
        for item in group:
            candidates = item if isinstance(item, list) else (item,)
            for s in candidates:
                idx = getattr(s, "index", None)
                if isinstance(idx, int):
                    return idx
        return 0

    out = sorted(collected.values(), key=_key)[:target]
    logger.info(
        "fully-async rollout %d: done in %.1fs, queue_left=%d",
        rollout_id,
        time.time() - started,
        worker.queue_size(),
    )
    return out


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation: bool = False):
    if evaluation:
        raise ValueError("fully-async rollout doesn't support evaluation mode")
    return run(_generate_rollout_async(args, rollout_id, data_buffer))
