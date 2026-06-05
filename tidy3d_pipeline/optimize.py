
from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

from tidy3d_pipeline.models import OptimizationRecord, new_id
from tidy3d_pipeline.store import PicStore


def optimize_over_real_sweep(
    store: PicStore,
    component_id: str = "y_junction",
    param: str = "l_junction",
    metric_name: str = "insertion_loss_mean",
    study_id: Optional[str] = None,
    context_params: tuple[str, ...] = ("w7", "h_bend"),
) -> dict:
    runs = store.compare_runs(component_id, metric_name)
    groups: dict[tuple, list] = defaultdict(list)
    for _, r in runs.iterrows():
        g = r["geometry_snapshot"] or {}
        if param in g and isinstance(g[param], (int, float)):
            ctx = tuple(g.get(c) for c in context_params)
            groups[ctx].append((float(g[param]), float(r["metric_value"]), r["run_id"]))
    if not groups:
        raise ValueError(f"No runs of '{component_id}' carry numeric geometry param '{param}'. "
                         f"Available params vary by component; check geometry_snapshot.")

    def slice_rank(ctx):
        c = groups[ctx]
        return (len({p for p, _, _ in c}), -min(o for _, o, _ in c))

    context = max(groups, key=slice_rank)
    candidates = sorted(groups[context], key=lambda x: x[0])
    context_dict = {c: v for c, v in zip(context_params, context) if v is not None}
    study_id = study_id or new_id("study", component_id, param, str(context))

    best = math.inf
    best_param = None
    best_run = None
    baseline = None
    for i, (pval, obj, run_id) in enumerate(candidates):
        if baseline is None:
            baseline = obj
        is_best = obj < best
        if is_best:
            best, best_param, best_run = obj, pval, run_id
        store.add_optimization_record(OptimizationRecord(
            record_id=new_id("opt", study_id, str(i)),
            study_id=study_id, component_id=component_id, iteration=i,
            parameters={param: pval, **context_dict}, objective_name=metric_name,
            objective_value=obj, best_so_far=best, is_best=is_best, run_id=run_id,
        ))

    return {
        "study_id": study_id, "param": param, "context": context_dict,
        "best_param": best_param, "best_objective": best,
        "baseline_objective": baseline,
        "improvement_db": (baseline - best) if baseline is not None else None,
        "best_run_id": best_run, "n_candidates": len(candidates),
    }
