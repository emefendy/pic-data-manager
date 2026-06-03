"""Pure data + analytics layer for the real-data dashboard (app_real.py).

No Streamlit here, so these functions are unit-testable headless. Two domains:

* Tidy3D PIC simulations — read REAL FDTD designs from pic_data.db
  (designer in {'sweep','simulation_import'}), with geometry from sim_params
  and the transmission/loss spectra from data/spectra/*.json.
* ScreeningDatasetV2 — the cascaded-MZI lab data, via the `screening` package's
  DuckDB+Parquet+Zarr store (built on demand from the raw dataset).

Shared analytics (correlations, a small predictive model, anomaly flags) work on
either domain's numeric feature frame.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "pic_data.db"
SPECTRA_DIR = ROOT / "data" / "spectra"
SCREEN_ROOT = ROOT / "data" / "screening"

# Designs we treat as REAL Tidy3D simulation output (vs synthetic seed data).
REAL_DESIGNERS = ("sweep", "simulation_import")

# sim_params keys that are OUTCOMES, not geometry — excluded from model features.
_OUTCOME_KEYS = ("loss", "bandwidth", "improvement")


# ===================================================================== #
# Tidy3D PIC simulation domain
# ===================================================================== #
def load_tidy3d_designs(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Real FDTD designs only, with sim_params JSON expanded into columns."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        placeholders = ",".join("?" * len(REAL_DESIGNERS))
        df = pd.read_sql_query(
            f"SELECT * FROM designs WHERE designer IN ({placeholders})",
            con, params=REAL_DESIGNERS,
        )
    finally:
        con.close()

    # expand sim_params JSON into geom_* columns
    geom_rows = []
    for s in df.get("sim_params", pd.Series([None] * len(df))):
        try:
            geom_rows.append(json.loads(s) if isinstance(s, str) else {})
        except Exception:
            geom_rows.append({})
    geom = pd.json_normalize(geom_rows).add_prefix("geom_")
    return pd.concat([df.reset_index(drop=True), geom.reset_index(drop=True)], axis=1)


def tidy3d_geometry_features(df: pd.DataFrame) -> list[str]:
    """Numeric geometry feature columns (geom_* minus outcome-like keys)."""
    feats = []
    for c in df.columns:
        if not c.startswith("geom_"):
            continue
        key = c[len("geom_"):].lower()
        if any(tok in key for tok in _OUTCOME_KEYS):
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any():
            feats.append(c)
    return feats


def load_spectrum(design_id: str, spectra_dir: Path = SPECTRA_DIR) -> Optional[dict]:
    p = Path(spectra_dir) / f"{design_id}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def spectrum_loss_curves(spec: dict) -> tuple[Optional[np.ndarray], dict[str, np.ndarray]]:
    """Return (wavelength_um, {curve_name: loss_db_array}) from a spectrum dict.

    Handles both Y-junction (insertion_loss_db) and Euler-bend
    (loss_euler_db / loss_circular_db) spectrum formats.
    """
    wl = None
    for k in ("wavelengths_um", "wavelength_um", "ldas"):
        if k in spec:
            wl = np.asarray(spec[k], dtype=float)
            break
    curves = {k: np.asarray(v, dtype=float) for k, v in spec.items()
              if isinstance(v, list) and "loss" in k.lower()}
    return wl, curves


# ===================================================================== #
# ScreeningDatasetV2 domain
# ===================================================================== #
def ensure_screening_store(dataset: Optional[str] = None,
                           root: Path = SCREEN_ROOT,
                           rebuild: bool = False):
    """Build the screening store from the raw dataset if missing; return root.

    Returns (root_path, status_str). status is 'ready', 'built', or an error msg.
    """
    db = Path(root) / "screening.duckdb"
    if db.exists() and not rebuild:
        return root, "ready"
    try:
        from screening.config import resolve_dataset
        from screening.pipeline import run_pipeline
        ds = resolve_dataset(dataset)
        run_pipeline(dataset=str(ds), root=str(root),
                     figdir=str(Path(root) / "figures"), reset=True)
        return root, "built"
    except Exception as e:  # dataset not found or build failed
        return root, f"error: {e}"


def open_screening_store(root: Path = SCREEN_ROOT):
    """Open the screening store read-only (no lock). Caller closes it."""
    from screening.store import ScreeningStore
    return ScreeningStore(root, read_only=True)


# ===================================================================== #
# Shared analytics (work on any numeric feature frame)
# ===================================================================== #
def build_feature_matrix(df: pd.DataFrame, feature_cols: list[str],
                         categorical: Optional[list[str]] = None) -> pd.DataFrame:
    """Numeric matrix: selected numeric cols (median-imputed) + one-hot categoricals."""
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    if categorical:
        for c in categorical:
            if c in df.columns and df[c].nunique() > 1:
                dummies = pd.get_dummies(df[c].astype(str), prefix=c)
                X = pd.concat([X, dummies], axis=1)
    return X.fillna(0.0)


def correlations(df: pd.DataFrame, feature_cols: list[str], target: str) -> pd.Series:
    """Pearson r of each numeric feature with the target (sorted by |r|)."""
    out = {}
    y = pd.to_numeric(df[target], errors="coerce")
    for c in feature_cols:
        x = pd.to_numeric(df[c], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() >= 3 and x[mask].std() > 0 and y[mask].std() > 0:
            out[c] = float(np.corrcoef(x[mask], y[mask])[0, 1])
    s = pd.Series(out)
    return s.reindex(s.abs().sort_values(ascending=False).index)


def fit_predictive_model(X: pd.DataFrame, y: pd.Series) -> dict:
    """Cross-validated RandomForest regression. Returns preds, metrics, importances."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.metrics import mean_absolute_error, r2_score

    y = pd.to_numeric(y, errors="coerce")
    mask = y.notna() & X.notna().all(axis=1)
    Xv, yv = X[mask].to_numpy(dtype=float), y[mask].to_numpy(dtype=float)
    n = len(yv)
    if n < 8:
        return {"ok": False, "reason": f"only {n} usable rows (need >= 8)"}

    model = RandomForestRegressor(n_estimators=300, random_state=0)
    cv = KFold(n_splits=min(5, n), shuffle=True, random_state=0)
    preds = cross_val_predict(model, Xv, yv, cv=cv)
    model.fit(Xv, yv)
    return {
        "ok": True, "n": n, "y_true": yv, "y_pred": preds,
        "r2": float(r2_score(yv, preds)), "mae": float(mean_absolute_error(yv, preds)),
        "importances": pd.Series(model.feature_importances_, index=X.columns)
                         .sort_values(ascending=False),
    }


def anomaly_flags(X: pd.DataFrame, contamination: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """IsolationForest anomaly labels (1=anomaly) + scores (higher=more anomalous)."""
    from sklearn.ensemble import IsolationForest
    Xv = X.fillna(X.median(numeric_only=True)).fillna(0.0).to_numpy(dtype=float)
    iso = IsolationForest(contamination=contamination, random_state=0)
    raw = iso.fit_predict(Xv)            # -1 anomaly, 1 normal
    score = -iso.score_samples(Xv)        # higher = more anomalous
    return (raw == -1).astype(int), score
