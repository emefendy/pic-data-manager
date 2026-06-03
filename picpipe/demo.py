"""End-to-end demo on REAL Tidy3D data: ingest -> query -> analyze -> optimize -> figures.

Single command:  python -m picpipe.cli demo
Reads this project's real FDTD sweeps (pic_data.db + data/spectra/*.json), with
no synthetic data anywhere.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from picpipe.components import EULER_BEND, Y_JUNCTION
from picpipe.ingest import ingest_real_dataset
from picpipe.optimize import optimize_over_real_sweep
from picpipe.store import PicStore


def _il(t):
    return -10.0 * np.log10(np.clip(t, 1e-12, None))


def run_demo(db_path: str = "pic_data.db", spectra_dir: str = "data/spectra",
             root: str = "data/pipeline", figdir: str = "data/pipeline/figures",
             reset: bool = True) -> dict:
    if reset and Path(root).exists():
        shutil.rmtree(root, ignore_errors=True)
    Path(figdir).mkdir(parents=True, exist_ok=True)
    store = PicStore(root)

    # 1) ingest every real design ----------------------------------------------
    run_ids = ingest_real_dataset(store, db_path, spectra_dir)

    # 2) optimization over the real Y-junction sweep ---------------------------
    opt = optimize_over_real_sweep(store, component_id="y_junction", param="l_junction")

    figs = {}

    # 3a) insertion loss vs wavelength — one Y-junction + one Euler bend --------
    fig, ax = plt.subplots(figsize=(7, 4))
    for comp, label in [(Y_JUNCTION, "Y-junction"), (EULER_BEND, "Euler bend")]:
        runs = store.runs_for_component(comp.component_id)
        if len(runs):
            sp = store.read_spectrum(runs.iloc[0]["run_id"])
            ax.plot(sp["wavelength_um"] * 1000, _il(sp["transmission"]), label=label)
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Insertion loss (dB)")
    ax.set_title("Insertion loss vs wavelength (real FDTD)"); ax.legend(); ax.grid(alpha=0.3)
    p = Path(figdir) / "insertion_loss_vs_wavelength.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); figs["spectra"] = str(p)

    # 3b) variant comparison — Y-junction mean IL vs l_junction ----------------
    cr = store.compare_runs("y_junction", "insertion_loss_mean")
    cr = cr.assign(l_junction=[(g or {}).get("l_junction") for g in cr["geometry_snapshot"]]).dropna(subset=["l_junction"])
    if len(cr):
        cr = cr.sort_values("l_junction")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(cr["l_junction"], cr["metric_value"], "o-")
        ax.set_xlabel("Junction length l_junction (µm)"); ax.set_ylabel("Mean insertion loss (dB)")
        ax.set_title("Y-junction variant comparison (real sweep)"); ax.grid(alpha=0.3)
        p = Path(figdir) / "variant_comparison.png"
        fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); figs["variants"] = str(p)

    # 3c) cross-component loss distribution ------------------------------------
    data, labels = [], []
    for comp, lab in [(Y_JUNCTION, "Y-junction"), (EULER_BEND, "Euler bend")]:
        m = store.metrics(component_id=comp.component_id, metric_name="insertion_loss_mean")
        if len(m):
            data.append(m["metric_value"].to_numpy()); labels.append(lab)
    if data:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.boxplot(data, labels=labels, showmeans=True)
        ax.set_ylabel("Mean insertion loss (dB)")
        ax.set_title("Insertion loss by component (real FDTD)"); ax.grid(alpha=0.3, axis="y")
        p = Path(figdir) / "component_comparison.png"
        fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); figs["components"] = str(p)

    # 3d) optimization progress -------------------------------------------------
    hist = store.optimization_history(opt["study_id"])
    if len(hist):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(hist["iteration"], hist["objective_value"], "o-", label="candidate")
        ax.plot(hist["iteration"], hist["best_so_far"], "s--", label="best so far")
        ax.set_xlabel("Iteration (sweep order by l_junction)")
        ax.set_ylabel("Mean insertion loss (dB)")
        ax.set_title("Y-junction optimization progress"); ax.legend(); ax.grid(alpha=0.3)
        p = Path(figdir) / "optimization_progress.png"
        fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); figs["optimization"] = str(p)

    # 4) summary table ----------------------------------------------------------
    summary = store.summary_table()
    summary_path = Path(figdir) / "summary_table.csv"
    summary.to_csv(summary_path, index=False)

    store.close()
    return {"root": root, "n_runs": len(run_ids), "figures": figs,
            "summary": summary, "summary_csv": str(summary_path), "optimization": opt}


if __name__ == "__main__":
    out = run_demo()
    print(f"Ingested {out['n_runs']} real Tidy3D runs.\n")
    print("=== SUMMARY (per component / metric) ===")
    print(out["summary"].to_string(index=False))
    print("\n=== OPTIMIZATION (Y-junction, real sweep) ===")
    for k, v in out["optimization"].items():
        print(f"  {k}: {v}")
    print("\nFigures:")
    for k, v in out["figures"].items():
        print(f"  {k}: {v}")
