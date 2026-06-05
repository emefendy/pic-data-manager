import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

PARAMS = ("w7", "h_bend", "l_junction")
DEFAULT_BOUNDS = {"w7": (1.2, 1.6), "h_bend": (1.5, 2.5), "l_junction": (1.5, 3.0)}


def design_id_for(p):
    return ("D_YJ_OPT_w{:03d}h{:03d}l{:03d}"
            .format(round(p["w7"] * 100), round(p["h_bend"] * 100),
                    round(p["l_junction"] * 100)))


def cached_objective(design_id):
    path = ROOT / "data" / "designs.json"
    if not path.exists():
        return None
    for r in json.loads(path.read_text()):
        if r["design_id"] == design_id:
            return float(r["simulated_loss_db"])
    return None


def seed_points():
    path = ROOT / "data" / "designs.json"
    if not path.exists():
        return []
    out = []
    for r in json.loads(path.read_text()):
        if r.get("device_type") != "y_junction":
            continue
        try:
            sp = json.loads(r.get("sim_params") or "{}")
        except Exception:
            continue
        if all(isinstance(sp.get(k), (int, float)) for k in PARAMS):
            p = {k: round(float(sp[k]), 3) for k in PARAMS}
            out.append((p, float(r["simulated_loss_db"]), r["design_id"]))
    return out


def propose_next(X_obs, y_obs, bounds, rng):
    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    X = np.asarray([[p[k] for k in PARAMS] for p in X_obs], dtype=float)
    y = np.asarray(y_obs, dtype=float)
    kernel = (ConstantKernel(1.0)
              * Matern(length_scale=[0.15, 0.4, 0.4], nu=2.5,
                       length_scale_bounds=[(0.05, 1.0), (0.2, 2.0), (0.25, 2.0)])
              + WhiteKernel(1e-6, noise_level_bounds=(1e-12, 1e-2)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=5, random_state=0)
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(X, y)

    lo = np.array([bounds[k][0] for k in PARAMS])
    hi = np.array([bounds[k][1] for k in PARAMS])
    cand = lo + (hi - lo) * rng.random((4096, len(PARAMS)))
    mu, sd = gp.predict(cand, return_std=True)
    best = y.min()
    z = (best - mu) / np.maximum(sd, 1e-12)
    ei = (best - mu) * norm.cdf(z) + sd * norm.pdf(z)
    scale = hi - lo
    for x in X:
        ei[np.linalg.norm((cand - x) / scale, axis=1) < 0.02] = -np.inf
    p = cand[int(np.argmax(ei))]
    return {k: round(float(v), 3) for k, v in zip(PARAMS, p)}


def make_evaluator(state):
    import tidy3d.web as web

    from sweep_yjunction import HDF5_DIR, T as THICK, W1, build_simulation, extract_and_ingest

    def evaluate(p):
        design_id = design_id_for(p)
        cached = cached_objective(design_id)
        if cached is not None:
            print(f"  {design_id} cached -> {cached:.4f} dB")
            return cached, design_id
        if state["evals"] >= state["max_evals"]:
            raise RuntimeError(f"budget of {state['max_evals']} new simulations exhausted")
        state["evals"] += 1
        print(f"  {design_id} (w7={p['w7']}, h_bend={p['h_bend']}, "
              f"l_junction={p['l_junction']}) submitting FDTD job "
              f"[{state['evals']}/{state['max_evals']}]...")
        sim, freqs = build_simulation(p["w7"], p["h_bend"], p["l_junction"])
        job = web.Job(simulation=sim, task_name=design_id, verbose=False)
        sim_data = job.run(path=str(HDF5_DIR / f"{design_id}.hdf5"))
        params = {"w1": W1, "t": THICK, **p}
        extract_and_ingest(design_id, sim_data, freqs, params)
        obj = cached_objective(design_id)
        print(f"  {design_id} -> {obj:.4f} dB")
        return obj, design_id

    return evaluate


def ingest_new_designs(design_ids):
    from build_manifest import build

    from tidy3d_pipeline.ingest import ingest_tidy3d_design
    from tidy3d_pipeline.store import PicStore

    build()
    import pandas as pd
    mf = pd.read_parquet(ROOT / "data" / "tidy3d_raw" / "manifest.parquet")
    run_ids = {}
    with PicStore(str(ROOT / "data" / "tidy3d_processed")) as store:
        for _, row in mf[mf["design_id"].isin(design_ids)].iterrows():
            rid = ingest_tidy3d_design(store, row.to_dict(),
                                       ROOT / "data" / "tidy3d_raw")
            if rid:
                run_ids[row["design_id"]] = rid
    return run_ids


def record_study(history_new, run_ids, seed_best, n_seeds):
    from tidy3d_pipeline.models import OptimizationRecord, new_id
    from tidy3d_pipeline.store import PicStore

    study_id = new_id("study", "y_junction", "bo3d", str(len(history_new)))
    best = seed_best
    with PicStore(str(ROOT / "data" / "tidy3d_processed")) as store:
        for i, (p, obj, did) in enumerate(history_new):
            is_best = obj < best
            best = min(best, obj)
            store.add_optimization_record(OptimizationRecord(
                record_id=new_id("opt", study_id, str(i)),
                study_id=study_id, component_id="y_junction", iteration=i,
                parameters={**p, "n_seeds": n_seeds, "seed_best_db": seed_best},
                objective_name="insertion_loss_mean", objective_value=obj,
                best_so_far=best, is_best=is_best,
                run_id=run_ids.get(did)))
    return study_id


def main():
    p = argparse.ArgumentParser(
        description="Closed-loop 3-D Bayesian optimization of (w7, h_bend, "
                    "l_junction): a Gaussian process seeded with every existing "
                    "sweep design proposes new parameter combinations; each "
                    "proposal runs a REAL Tidy3D FDTD job (costs FlexCredits).")
    p.add_argument("--max-evals", type=int, default=6,
                   help="hard cap on NEW simulations")
    for k in PARAMS:
        lo, hi = DEFAULT_BOUNDS[k]
        p.add_argument(f"--{k.replace('_','-')}-lo", type=float, default=lo)
        p.add_argument(f"--{k.replace('_','-')}-hi", type=float, default=hi)
    args = p.parse_args()
    bounds = {k: (getattr(args, f"{k}_lo"), getattr(args, f"{k}_hi")) for k in PARAMS}

    seeds = seed_points()
    if not seeds:
        print("No existing designs found to seed the optimizer."); return 1
    seed_best = min(o for _, o, _ in seeds)
    print(f"Seeded the GP with {len(seeds)} existing sweep designs (free) — "
          f"best existing: {seed_best:.4f} dB")
    print("Optimizing all three swept parameters simultaneously:")
    for k in PARAMS:
        print(f"  {k} in [{bounds[k][0]}, {bounds[k][1]}]")
    print(f"Fixed: w1 = 0.5 um, t = 0.22 um, band 1.5-1.6 um. "
          f"Budget: {args.max_evals} new FDTD jobs.\n")

    state = {"evals": 0, "max_evals": args.max_evals}
    evaluate = make_evaluator(state)
    rng = np.random.default_rng(0)
    X_obs = [pt for pt, _, _ in seeds]
    y_obs = [o for _, o, _ in seeds]
    history_new = []

    try:
        while state["evals"] < state["max_evals"]:
            prop = propose_next(X_obs, y_obs, bounds, rng)
            print(f"  GP proposes {prop}")
            obj, did = evaluate(prop)
            X_obs.append(prop); y_obs.append(obj)
            history_new.append((prop, obj, did))
    except RuntimeError as e:
        print(f"\nStopping: {e}")

    if not history_new:
        print("No new evaluations completed."); return 1

    print("\nRebuilding manifest + ingesting new designs into the store...")
    run_ids = ingest_new_designs([did for _, _, did in history_new])
    study_id = record_study(history_new, run_ids, seed_best, len(seeds))

    best_p, best_obj, best_id = min(history_new, key=lambda h: h[1])
    print(f"\nStudy {study_id}: {len(history_new)} new FDTD evaluations "
          f"(+{len(seeds)} free seeds informing the GP)")
    print(f"Best new: {best_p} -> {best_obj:.4f} dB ({best_id})")
    print(f"Best existing (seed): {seed_best:.4f} dB")
    if best_obj < seed_best:
        print(f"Improvement over every pre-existing design: "
              f"{seed_best - best_obj:.4f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
