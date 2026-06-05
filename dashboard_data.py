
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SPECTRA_DIR = ROOT / "data" / "tidy3d_raw"
MANIFEST = SPECTRA_DIR / "manifest.parquet"
SCREEN_ROOT = ROOT / "data" / "reallabdata_processed"

_OUTCOME_KEYS = ("loss", "bandwidth", "improvement")

PARAM_LABELS = {
    "l_junction": "junction (taper) length l_junction (µm)",
    "w1": "input waveguide width w1 (µm)",
    "w7": "junction output width w7 (µm)",
    "t": "silicon thickness t (µm)",
    "h_bend": "S-bend vertical offset h_bend (µm)",
    "R_eff": "effective bend radius R_eff (µm)",
    "A": "Euler curvature parameter A",
    "w": "waveguide width w (µm)",
}


def param_label(p: str) -> str:
    return PARAM_LABELS.get(p.replace("geom_", ""), p)


def load_tidy3d_designs(manifest: Path = MANIFEST) -> pd.DataFrame:
    if not Path(manifest).exists():
        try:
            import sys
            sys.path.insert(0, str(ROOT / "scripts"))
            from build_manifest import build
            build()
        except Exception:
            return pd.DataFrame()
    df = pd.read_parquet(manifest)

    geom_rows = []
    for s in df.get("sim_params", pd.Series([None] * len(df))):
        try:
            geom_rows.append(json.loads(s) if isinstance(s, str) else {})
        except Exception:
            geom_rows.append({})
    geom = pd.json_normalize(geom_rows).add_prefix("geom_")
    return pd.concat([df.reset_index(drop=True), geom.reset_index(drop=True)], axis=1)


def tidy3d_geometry_features(df: pd.DataFrame) -> list[str]:
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
    wl = None
    for k in ("wavelengths_um", "wavelength_um", "ldas"):
        if k in spec:
            wl = np.asarray(spec[k], dtype=float)
            break
    curves = {k: np.asarray(v, dtype=float) for k, v in spec.items()
              if isinstance(v, list) and "loss" in k.lower()}
    return wl, curves


def ensure_screening_store(dataset: Optional[str] = None,
                           root: Path = SCREEN_ROOT,
                           rebuild: bool = False):
    db = Path(root) / "reallabdata.duckdb"
    if db.exists() and not rebuild:
        return root, "ready"
    try:
        from reallabdata_pipeline.config import resolve_dataset
        from reallabdata_pipeline.pipeline import run_pipeline
        ds = resolve_dataset(dataset)
        run_pipeline(dataset=str(ds), root=str(root),
                     figdir=str(Path(root) / "figures"), reset=True)
        return root, "built"
    except Exception as e:
        return root, f"error: {e}"


def open_screening_store(root: Path = SCREEN_ROOT):
    from reallabdata_pipeline.store import ScreeningStore
    return ScreeningStore(root, read_only=True)


def build_feature_matrix(df: pd.DataFrame, feature_cols: list[str],
                         categorical: Optional[list[str]] = None) -> pd.DataFrame:
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    if categorical:
        for c in categorical:
            if c in df.columns and df[c].nunique() > 1:
                dummies = pd.get_dummies(df[c].astype(str), prefix=c)
                X = pd.concat([X, dummies], axis=1)
    return X.fillna(0.0)


def correlations(df: pd.DataFrame, feature_cols: list[str], target: str) -> pd.Series:
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
        "row_index": np.where(mask.to_numpy())[0],
        "r2": float(r2_score(yv, preds)), "mae": float(mean_absolute_error(yv, preds)),
        "importances": pd.Series(model.feature_importances_, index=X.columns)
                         .sort_values(ascending=False),
    }


def shap_attributions(X: pd.DataFrame, y: pd.Series) -> dict:
    from sklearn.ensemble import RandomForestRegressor

    y = pd.to_numeric(y, errors="coerce")
    mask = y.notna() & X.notna().all(axis=1)
    row_index = np.where(mask.to_numpy())[0]
    Xv = X[mask]
    if len(row_index) < 8 or Xv.shape[1] == 0:
        return {"ok": False, "reason": f"need >= 8 rows (have {len(row_index)})"}
    try:
        import shap
    except Exception:
        return {"ok": False, "reason": "shap is not installed (pip install shap)"}

    model = RandomForestRegressor(n_estimators=300, random_state=0)
    model.fit(Xv.to_numpy(dtype=float), y[mask].to_numpy(dtype=float))
    expl = shap.TreeExplainer(model)
    sv = np.asarray(expl.shap_values(Xv.to_numpy(dtype=float)))
    base = float(np.asarray(expl.expected_value).ravel()[0])
    return {"ok": True, "shap": sv, "X": Xv.reset_index(drop=True),
            "features": list(Xv.columns), "base": base, "row_index": row_index}


PIPELINE_ROOT = ROOT / "data" / "tidy3d_processed"


def ensure_tidy3d_pipeline_store(root: Path = PIPELINE_ROOT, rebuild: bool = False):
    db = Path(root) / "tidy3d.duckdb"
    if db.exists() and not rebuild:
        return root, "ready"
    try:
        from tidy3d_pipeline.demo import run_demo
        run_demo(manifest=str(MANIFEST), spectra_dir=str(ROOT / "data" / "tidy3d_raw"),
                 root=str(root), figdir=str(Path(root) / "figures"), reset=True)
        return root, "built"
    except Exception as e:
        return root, f"error: {e}"


def duckdb_overview(db_path) -> pd.DataFrame:
    import duckdb
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        rows = [{"table": t, "rows": con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]}
                for t in tables]
    finally:
        con.close()
    return pd.DataFrame(rows)


def duckdb_preview(db_path, table: str, n: int = 20) -> pd.DataFrame:
    import duckdb
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(f'SELECT * FROM "{table}" LIMIT {int(n)}').df()
    finally:
        con.close()


def load_component_record(component_type: str) -> Optional[dict]:
    import duckdb

    db = PIPELINE_ROOT / "tidy3d.duckdb"
    if not db.exists():
        return None
    con = duckdb.connect(str(db), read_only=True)
    try:
        df = con.execute("SELECT * FROM components WHERE type = ?",
                         [component_type]).df()
    finally:
        con.close()
    if df.empty:
        return None
    rec = df.iloc[0].to_dict()
    for k in ("materials", "geometry", "expected_metrics"):
        if isinstance(rec.get(k), str):
            try:
                rec[k] = json.loads(rec[k])
            except Exception:
                pass
    return rec


def load_optimization_records(component_id: str) -> pd.DataFrame:
    import duckdb

    db = PIPELINE_ROOT / "tidy3d.duckdb"
    if not db.exists():
        return pd.DataFrame()
    con = duckdb.connect(str(db), read_only=True)
    try:
        df = con.execute(
            "SELECT study_id, iteration, parameters, objective_name, "
            "objective_value, best_so_far, is_best, run_id, created_at "
            "FROM optimization_records WHERE component_id = ? "
            "ORDER BY created_at, iteration", [component_id]).df()
    finally:
        con.close()
    if len(df):
        df["parameters"] = df["parameters"].map(
            lambda s: json.loads(s) if isinstance(s, str) else (s or {}))
    return df


def fit_two_stage_fringe(v_dc, pd_mv, n_starts: int = 60, seed: int = 0) -> dict:
    from scipy.optimize import curve_fit

    v = np.asarray(v_dc, dtype=float)
    p = np.asarray(pd_mv, dtype=float)

    def model(V, P0, A, Vc1, Vpi1, Vc2, Vpi2):
        c1 = np.cos(np.pi * (V - Vc1) / (2 * Vpi1)) ** 2
        c2 = np.cos(np.pi * (V - Vc2) / (2 * Vpi2)) ** 2
        return P0 + A * c1 * c2

    span = float(v.max() - v.min())
    lo = [0.0, 1.0, v.min() - 2, 0.8, v.min() - 2, 0.8]
    hi = [max(10.0, p.min() * 2 + 1), 4 * p.max(), v.max() + 2, span * 1.2,
          v.max() + 2, span * 1.2]
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_starts):
        guess = [p.min(), p.max() - p.min(),
                 rng.uniform(v.min(), v.max()), rng.uniform(1.0, span * 0.8),
                 rng.uniform(v.min(), v.max()), rng.uniform(1.0, span * 0.8)]
        try:
            popt, _ = curve_fit(model, v, p, p0=guess, bounds=(lo, hi), maxfev=20000)
            sse = float(((model(v, *popt) - p) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, popt)
        except Exception:
            continue
    if best is None:
        return {"ok": False, "reason": "no fit converged"}

    sse, popt = best
    P0, A = float(popt[0]), float(popt[1])
    r2 = 1.0 - sse / float(((p - p.mean()) ** 2).sum())

    stages = []
    for vc, vpi in sorted([(float(popt[2]), float(popt[3])),
                           (float(popt[4]), float(popt[5]))], key=lambda s: s[1]):
        ks = np.arange(-int(span / vpi) - 2, int(span / vpi) + 3)
        nulls = vc + (2 * ks + 1) * vpi
        nulls = [float(n) for n in nulls if v.min() - 1e-9 <= n <= v.max() + 1e-9]
        stages.append({"v_pi": vpi, "v_peak": vc, "nulls": nulls})

    grid = np.linspace(v.min(), v.max(), 600)
    stage_curves = [A * np.cos(np.pi * (grid - s["v_peak"]) / (2 * s["v_pi"])) ** 2
                    for s in stages]
    return {"ok": True, "P0": P0, "A": A, "r2": r2, "stages": stages,
            "v": v, "p": p, "grid": grid, "fit": model(grid, *popt),
            "stage_curves": stage_curves}


DUCKDB_SCHEMA_HINT = """
DuckDB store for a PIC simulation pipeline. Tables:

components(component_id, type, name, library, platform, geometry /*JSON*/, expected_metrics /*JSON*/)
  -- type is 'y_junction' or 'euler_bend'.
simulation_runs(run_id, component_id, tool, tool_version, wl_min_um, wl_max_um,
  n_wavelengths, boundary_conditions, mesh_summary, geometry_snapshot /*JSON, e.g. l_junction,w1,w7,R_eff,A*/,
  task_id, status)
analysis_results(result_id, run_id, component_id, metric_name, metric_value, units)
  -- metric_name in ('insertion_loss_mean','insertion_loss_min','insertion_loss_at_center','bandwidth_3db').
optimization_records(record_id, study_id, component_id, iteration, parameters /*JSON*/,
  objective_name, objective_value, best_so_far, is_best, run_id)
raw_artifacts(artifact_id, run_id, kind, path, format, checksum_sha256, size_bytes)

Notes:
- Extract JSON geometry with json_extract(geometry_snapshot, '$.l_junction').
- The per-run transmission spectra are Parquet files in data/tidy3d_processed/arrays/<run_id>.parquet,
  readable with read_parquet('data/tidy3d_processed/arrays/<run_id>.parquet') (columns: wavelength_um, transmission).
"""


def nl_to_duckdb_sql(question: str, api_key: str, schema: Optional[str] = None) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    schema = schema or DUCKDB_SCHEMA_HINT
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=512,
        system=("You are a DuckDB expert for a photonics data store. Convert the "
                "user's question into a single correct DuckDB SELECT query. Return ONLY raw SQL "
                "— no markdown fences, no prose. SELECT only (read-only). LIMIT 200 unless asked "
                f"for all rows.\n\nSchema:\n{schema}"),
        messages=[{"role": "user", "content": question}],
    )
    sql = msg.content[0].text.strip()
    if sql.startswith("```"):
        sql = sql.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return sql


def _tidy3d_store():
    from tidy3d_pipeline.store import PicStore
    return PicStore(PIPELINE_ROOT, read_only=True)


def geometry_param_names() -> list[str]:
    df = load_tidy3d_designs()
    return [c[len("geom_"):] for c in tidy3d_geometry_features(df)]


REALLAB_SCHEMA_HINT = """
DuckDB store for real lab measurements of a cascaded Mach-Zehnder modulator. Tables:

devices(device_id, name, type, pd_bandwidth_hz, ac_coupling_cutoff_hz, n_stages)
measurements(measurement_id, device_id, kind /* 'static_sweep' | 'fra' | 'capture' */,
  tag /* e.g. rep30kHz_e10ns_sym1 */, params /*JSON: rep_hz, edge_ns, vlbl, V_pp, V_DC*/,
  n_points, n_traces)
analysis_results(result_id, measurement_id, device_id, metric_name, metric_value, units,
  "group" /* 'metric' = device-level, 'feature' = per-capture */)
  -- device metrics: fra_bandwidth_3db, fra_dc_gain, static_V_null, static_V_pi_est,
  --                 static_extinction, static_visibility, static_PD_min, static_PD_max
  -- per-capture features: cap_swing_mV, cap_rise_time_ns, cap_overshoot_pct,
  --                       cap_baseline_mV, cap_top_mV, cap_trace_noise_mV
raw_artifacts(artifact_id, measurement_id, kind, path, format /* parquet | zarr */)
lab_logs(log_id, device_id, date, title, body, referenced_tags /*JSON*/)

Notes:
- Extract params with json_extract(params, '$.rep_hz') etc.
- "group" is a reserved word: always double-quote it.
"""


def run_duckdb_sql(db_path, sql: str) -> pd.DataFrame:
    import duckdb
    if not sql.lstrip().lower().startswith(("select", "with")):
        raise ValueError("Only read-only SELECT/WITH queries are allowed.")
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(sql).df()
    finally:
        con.close()


def interpret_result(question: str, sql: str, result_text: str, api_key: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=400,
        system=("You are a PIC data analyst. Interpret the query result concisely for an "
                "engineer — key numbers and the takeaway up front, plain prose, no bullet lists."),
        messages=[{"role": "user", "content":
                   f"Question: {question}\n\nSQL:\n{sql}\n\nResult:\n{result_text}"}],
    )
    return msg.content[0].text


def rel(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


