from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reallabdata_pipeline import ingest
from reallabdata_pipeline.config import resolve_dataset
from reallabdata_pipeline.store import ScreeningStore


def run_pipeline(dataset=None, root: str = "data/reallabdata_processed", figdir: str = "data/reallabdata_processed/figures",
                 reset: bool = True, capture_limit=None, store_full_traces: bool = True) -> dict:
    ds = resolve_dataset(dataset)
    if reset and Path(root).exists():
        shutil.rmtree(root, ignore_errors=True)
    Path(figdir).mkdir(parents=True, exist_ok=True)
    store = ScreeningStore(root)

    device_id = ingest.ingest_device_card(store, str(ds / "device_card.md"))
    static_mid = ingest.ingest_static_sweep(store, str(ds / "static_vdc_sweep.npz"))
    fra_mid = ingest.ingest_fra(store, str(ds / "fra_smallsignal.npz"))
    cap_ids = ingest.ingest_captures(store, str(ds / "captures"), limit=capture_limit,
                                     store_full_traces=store_full_traces)
    log_ids = ingest.ingest_lab_logs(store, str(ds / "lab_log_examples"))

    figs = {}

    sdf = store.read_table(static_mid)
    sm = {r["metric_name"]: r["metric_value"]
          for _, r in store.metrics(static_mid).iterrows()}
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sdf["V_DC_volts"], sdf["PD_mean_mV"], lw=1.5)
    if "static_V_null" in sm:
        ax.axvline(sm["static_V_null"], color="crimson", ls="--",
                   label=f"V_null = {sm['static_V_null']:.2f} V")
    ax.set_xlabel("Static bias V_DC (V)"); ax.set_ylabel("PD mean (mV)")
    ax.set_title("Static fringe — cascaded MZI"); ax.legend(); ax.grid(alpha=0.3)
    p = Path(figdir) / "static_fringe.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); figs["static_fringe"] = str(p)

    fdf = store.read_table(fra_mid)
    fm = {r["metric_name"]: r["metric_value"]
          for _, r in store.metrics(fra_mid).iterrows()}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
    a1.semilogx(fdf["freq_hz"], fdf["H_mag_db"]); a1.set_ylabel("|H| (dB)")
    if "fra_bandwidth_3db" in fm and np.isfinite(fm["fra_bandwidth_3db"]):
        a1.axvline(fm["fra_bandwidth_3db"] * 1e6, color="crimson", ls="--",
                   label=f"-3 dB = {fm['fra_bandwidth_3db']:.2f} MHz")
        a1.legend()
    a1.set_title("Small-signal Bode (drive→PD)"); a1.grid(alpha=0.3, which="both")
    a2.semilogx(fdf["freq_hz"], fdf["H_phase_deg"]); a2.set_ylabel("phase (deg)")
    a2.set_xlabel("Frequency (Hz)"); a2.grid(alpha=0.3, which="both")
    p = Path(figdir) / "bode.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); figs["bode"] = str(p)

    caps = store.captures()
    if not caps.empty:
        tag = caps.sort_values(["rep_hz", "edge_ns"]).iloc[0]["tag"]
        arrays, attrs = store.read_waveform(tag)
        t_us = arrays["time"] * 1e6
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(t_us, arrays["ch1_mean"] * 1e3, lw=1.2, label="PD (ch1, mean)")
        sd = None
        if "ch1_std" in arrays:
            sd = arrays["ch1_std"] * 1e3
        elif "ch1" in arrays and arrays["ch1"].ndim == 2:
            sd = arrays["ch1"].std(axis=0) * 1e3
        if sd is not None:
            ax.fill_between(t_us, arrays["ch1_mean"] * 1e3 - sd,
                            arrays["ch1_mean"] * 1e3 + sd, alpha=0.25, label="±1σ traces")
        ax.set_xlabel("Time (µs)"); ax.set_ylabel("PD (mV)")
        ax.set_title(f"Step capture {tag}"); ax.legend(); ax.grid(alpha=0.3)
        p = Path(figdir) / "capture_example.png"
        fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); figs["capture"] = str(p)

    feats = store.feature_table()
    if not feats.empty and "cap_swing_mV" in feats.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        for vlbl, grp in feats.groupby("vlbl"):
            g = grp.groupby("rep_hz")["cap_swing_mV"].mean().sort_index()
            ax.plot(g.index / 1e3, g.values, "o-", label=vlbl)
        ax.set_xlabel("Rep rate (kHz)"); ax.set_ylabel("Mean PD swing (mV)")
        ax.set_xscale("log"); ax.set_title("PD swing vs rep rate by drive variant")
        ax.legend(title="drive"); ax.grid(alpha=0.3, which="both")
        p = Path(figdir) / "swing_vs_reprate.png"
        fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); figs["swing"] = str(p)

    summary = store.device_metrics_summary()
    summary_path = Path(figdir) / "device_metrics.csv"
    summary.to_csv(summary_path, index=False)
    feats_path = Path(figdir) / "capture_features.csv"
    if not feats.empty:
        feats.to_csv(feats_path, index=False)

    result = {
        "dataset": str(ds), "root": root, "figures": figs,
        "n_captures": len(cap_ids), "n_lab_logs": len(log_ids),
        "device_metrics": summary, "feature_table": feats,
        "summary_csv": str(summary_path),
    }
    store.close()
    return result


if __name__ == "__main__":
    out = run_pipeline()
    print(f"dataset: {out['dataset']}")
    print(f"ingested {out['n_captures']} captures, {out['n_lab_logs']} lab logs")
    print("\n=== DEVICE METRICS ===")
    print(out["device_metrics"].to_string(index=False))
    print("\nFigures:")
    for k, v in out["figures"].items():
        print(f"  {k}: {v}")
