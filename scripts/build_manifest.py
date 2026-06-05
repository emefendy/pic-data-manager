
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DESIGNS_JSON = ROOT / "data" / "designs.json"
MANIFEST = ROOT / "data" / "tidy3d_raw" / "manifest.parquet"

REAL_DESIGNERS = ("sweep", "simulation_import")
KEEP = ["design_id", "device_type", "designer", "simulated_loss_db",
        "simulated_bandwidth_ghz", "simulation_file", "sim_params"]


def build() -> Path:
    records = json.loads(DESIGNS_JSON.read_text())
    df = pd.DataFrame(records)
    real = df[df["designer"].isin(REAL_DESIGNERS)].copy()
    for c in KEEP:
        if c not in real.columns:
            real[c] = None
    real = real[KEEP].reset_index(drop=True)
    real["sim_params"] = real["sim_params"].apply(
        lambda v: v if isinstance(v, str) else json.dumps(v or {}))
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    real.to_parquet(MANIFEST, index=False)
    print(f"Wrote {len(real)} real designs -> {MANIFEST.relative_to(ROOT)} "
          f"({real['device_type'].value_counts().to_dict()})")
    return MANIFEST


if __name__ == "__main__":
    build()
