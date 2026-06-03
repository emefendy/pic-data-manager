"""Stretch challenge — geometry optimization over the REAL Tidy3D sweep.

This project already ran a real FDTD parameter sweep of the Y-junction
(scripts/sweep_yjunction.py). The optimization here is a reproducible,
gradient-free parameter sweep over those *real* results: it walks the swept
geometry parameter in order, treats each design's measured mean insertion loss
as the objective, tracks the best-so-far, and records every candidate in the
pipeline as an OptimizationRecord (linked to its SimulationRun).

Objective: minimize ``insertion_loss_mean`` (dB). Single-metric, no penalties.
Constraint/simplification: candidates are drawn from the precomputed real FDTD
sweep grid (no new solver calls), so the "search" is over the existing grid —
honest and fully reproducible. To optimize on a finer/continuous grid, pass a
Tidy3D-backed evaluator that runs new jobs (see picpipe/sim/tidy3d_runner.py).
"""

from __future__ import annotations

import math
from typing import Optional

from picpipe.models import OptimizationRecord, new_id
from picpipe.store import PicStore


def optimize_over_real_sweep(
    store: PicStore,
    component_id: str = "y_junction",
    param: str = "l_junction",
    metric_name: str = "insertion_loss_mean",
    study_id: Optional[str] = None,
) -> dict:
    """Sweep the real designs of a component by ``param``; record progress.

    Returns a dict with study_id, best_param, best_objective, baseline_objective,
    improvement_db, best_run_id, n_candidates.
    """
    runs = store.compare_runs(component_id, metric_name)
    candidates = []
    for _, r in runs.iterrows():
        g = r["geometry_snapshot"] or {}
        if param in g and isinstance(g[param], (int, float)):
            candidates.append((float(g[param]), float(r["metric_value"]), r["run_id"]))
    if not candidates:
        raise ValueError(f"No runs of '{component_id}' carry numeric geometry param '{param}'. "
                         f"Available params vary by component; check geometry_snapshot.")

    candidates.sort(key=lambda x: x[0])  # walk the sweep in parameter order
    study_id = study_id or new_id("study", component_id, param)

    best = math.inf
    best_param = None
    best_run = None
    baseline = None
    for i, (pval, obj, run_id) in enumerate(candidates):
        if baseline is None:
            baseline = obj  # first candidate = initial design reference
        is_best = obj < best
        if is_best:
            best, best_param, best_run = obj, pval, run_id
        store.add_optimization_record(OptimizationRecord(
            record_id=new_id("opt", study_id, str(i)),
            study_id=study_id, component_id=component_id, iteration=i,
            parameters={param: pval}, objective_name=metric_name,
            objective_value=obj, best_so_far=best, is_best=is_best, run_id=run_id,
        ))

    return {
        "study_id": study_id, "param": param,
        "best_param": best_param, "best_objective": best,
        "baseline_objective": baseline,
        "improvement_db": (baseline - best) if baseline is not None else None,
        "best_run_id": best_run, "n_candidates": len(candidates),
    }
