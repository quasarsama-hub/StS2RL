"""Play many runs in parallel worker processes.

Each worker owns its own copy of the native simulator (handles are process-local) and
loads the value net from disk, keeping it until the trainer publishes a newer one.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import asdict
from pathlib import Path

from .agent import AgentConfig, RunRecord

_worker_model = None
_worker_model_key = None


def _load_model(path: str | None):
    global _worker_model, _worker_model_key
    if path is None:
        _worker_model, _worker_model_key = None, None
        return None
    key = (path, os.path.getmtime(path))
    if key != _worker_model_key:
        from .model import ValueNet

        _worker_model = ValueNet.load(path)
        _worker_model_key = key
    return _worker_model


def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)


def _play_one(task) -> RunRecord:
    model_path, seed, cfg_dict, agent_seed = task
    from .agent import Agent

    model = _load_model(model_path)
    agent = Agent(model, AgentConfig(**cfg_dict), seed=agent_seed)
    try:
        return agent.play(seed)
    except Exception as exc:  # one broken run must not take the batch down
        rec = RunRecord(seed=str(seed))
        rec.death_encounter = f"ERROR: {type(exc).__name__}: {exc}"
        return rec


class Pool:
    """Process pool that survives a worker dying (a native crash), unlike multiprocessing.Pool."""

    def __init__(self, workers: int):
        self.workers = max(1, workers)
        self._ex = self._make() if self.workers > 1 else None

    def _make(self):
        from concurrent.futures import ProcessPoolExecutor

        return ProcessPoolExecutor(self.workers, mp_context=mp.get_context("spawn"), initializer=_init_worker)

    def play(self, seeds, cfg: AgentConfig, model_path: Path | str | None, base_seed: int = 0):
        tasks = [(str(model_path) if model_path else None, s, asdict(cfg), base_seed + i) for i, s in enumerate(seeds)]
        if self._ex is None:
            _init_worker()
            yield from map(_play_one, tasks)
            return
        from concurrent.futures import as_completed
        from concurrent.futures.process import BrokenProcessPool

        pending = list(tasks)
        for attempt in range(2):
            futures = {self._ex.submit(_play_one, t): t for t in pending}
            pending = []
            for fut in as_completed(futures):
                try:
                    yield fut.result()
                except BrokenProcessPool:
                    pending.append(futures[fut])
                except Exception as exc:  # noqa: BLE001
                    rec = RunRecord(seed=str(futures[fut][1]))
                    rec.death_encounter = f"ERROR: {type(exc).__name__}: {exc}"
                    yield rec
            if not pending:
                return
            # A worker died and took the in-flight runs with it. Start over once.
            self._ex.shutdown(wait=False, cancel_futures=True)
            self._ex = self._make()
        for t in pending:
            rec = RunRecord(seed=str(t[1]))
            rec.death_encounter = "ERROR: worker process died"
            yield rec

    def close(self) -> None:
        if self._ex is not None:
            self._ex.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
