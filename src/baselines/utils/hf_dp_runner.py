from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.multiprocessing as mp


def _visible_gpu_ids(nprocs: int) -> List[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        ids = [x.strip() for x in visible.split(",") if x.strip()]
        if len(ids) < nprocs:
            raise RuntimeError(
                f"CUDA_VISIBLE_DEVICES has {len(ids)} GPUs but nprocs={nprocs}: "
                f"{visible}")
        return ids[:nprocs]
    return [str(i) for i in range(nprocs)]


def _temporary_parent() -> Path:
    configured = os.environ.get("KT_HF_DP_TMPDIR") or os.environ.get("KT_TMPDIR")
    if configured:
        parent = Path(configured).expanduser()
    else:
        parent = Path.cwd() / "tmp" / "hf_dp"
    parent.mkdir(parents=True, exist_ok=True)
    return parent


def _run_worker(rank: int, gpu_id: str, nprocs: int, worker_fn: Callable,
                questions: List[str], extras: Dict, tmp_dir: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = "cuda:0"
    slice_idx = list(range(rank, len(questions), nprocs))
    slice_qs = [questions[i] for i in slice_idx]
    out = worker_fn(rank=rank, device=device, questions=slice_qs,
                    indices=slice_idx, extras=extras)
    tmp_path = Path(tmp_dir)
    tmp_path.mkdir(parents=True, exist_ok=True)
    with open(tmp_path / f"rank{rank}.pkl", "wb") as f:
        pickle.dump(dict(indices=slice_idx, results=out), f)


def run_hf_dp(worker_fn: Callable, questions: List[str],
              nprocs: int = 8, extras: Optional[Dict] = None) -> List[Any]:
    extras = extras or {}
    gpu_ids = _visible_gpu_ids(nprocs)
    print(f"[hf_dp] using GPUs: {','.join(gpu_ids)}")
    ctx = mp.get_context("spawn")
    tmp_parent = _temporary_parent()
    with tempfile.TemporaryDirectory(prefix="hf_dp_", dir=str(tmp_parent)) as tmp_dir:
        print(f"[hf_dp] tmp={tmp_dir}")
        procs = []
        for rank in range(nprocs):
            p = ctx.Process(target=_run_worker,
                             args=(rank, gpu_ids[rank], nprocs, worker_fn,
                                   questions,
                                   extras, tmp_dir))
            p.start()
            procs.append(p)
        failures = []
        for rank, p in enumerate(procs):
            p.join()
            if p.exitcode != 0:
                failures.append((rank, p.exitcode))
        if failures:
            detail = ", ".join(f"rank{rank}=exit{code}"
                               for rank, code in failures)
            raise RuntimeError(f"hf_dp worker rank failed: {detail}")

        merged = [None] * len(questions)
        for rank in range(nprocs):
            part = Path(tmp_dir) / f"rank{rank}.pkl"
            if not part.exists():
                raise RuntimeError(f"hf_dp worker rank{rank} produced no output: "
                                   f"{part}")
            with open(part, "rb") as f:
                payload = pickle.load(f)
            for idx, r in zip(payload["indices"], payload["results"]):
                merged[idx] = r
        return merged
