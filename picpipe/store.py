"""Storage and query layer — DuckDB + Parquet + Zarr.

A three-part storage stack, each piece doing what it is best at:

* **DuckDB** is the metadata/provenance database and the query engine. It is an
  in-process analytical SQL engine (no server), a single portable file, and it
  can query Parquet directly — a natural fit for a data-engineering pipeline.
* **Parquet** holds the *tabular* array payloads (1-D transmission spectra,
  S-parameter tables). Columnar, compressed, language-agnostic.
* **Zarr** holds the *N-dimensional* array payloads (field-monitor cubes).

The relational tables never hold large arrays — they hold a `RawArtifact` row
with a path + SHA-256 checksum pointing at the Parquet or Zarr payload.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from picpipe.models import (
    AnalysisResult,
    Component,
    OptimizationRecord,
    RawArtifact,
    SimulationRun,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS components (
    component_id    VARCHAR PRIMARY KEY,
    type            VARCHAR NOT NULL,
    name            VARCHAR NOT NULL,
    library         VARCHAR NOT NULL,
    source_url      VARCHAR,
    function        VARCHAR,
    platform        VARCHAR,
    materials       VARCHAR,   -- json
    geometry        VARCHAR,   -- json
    expected_metrics VARCHAR,  -- json
    notes           VARCHAR,
    created_at      VARCHAR
);

CREATE TABLE IF NOT EXISTS simulation_runs (
    run_id          VARCHAR PRIMARY KEY,
    component_id    VARCHAR NOT NULL,
    tool            VARCHAR NOT NULL,
    tool_version    VARCHAR,
    is_synthetic    BOOLEAN NOT NULL DEFAULT FALSE,
    wl_min_um       DOUBLE,
    wl_max_um       DOUBLE,
    n_wavelengths   INTEGER,
    boundary_conditions VARCHAR,
    mesh_summary    VARCHAR,
    settings        VARCHAR,   -- json
    geometry_snapshot VARCHAR, -- json
    task_id         VARCHAR,
    status          VARCHAR,
    flex_credit_cost DOUBLE,
    created_at      VARCHAR
);

CREATE TABLE IF NOT EXISTS raw_artifacts (
    artifact_id     VARCHAR PRIMARY KEY,
    run_id          VARCHAR NOT NULL,
    kind            VARCHAR,
    path            VARCHAR,
    format          VARCHAR,
    checksum_sha256 VARCHAR,
    size_bytes      BIGINT,
    created_at      VARCHAR
);

CREATE TABLE IF NOT EXISTS analysis_results (
    result_id       VARCHAR PRIMARY KEY,
    run_id          VARCHAR NOT NULL,
    component_id    VARCHAR NOT NULL,
    metric_name     VARCHAR NOT NULL,
    metric_value    DOUBLE NOT NULL,
    units           VARCHAR,
    wavelength_um   DOUBLE,
    definition      VARCHAR,
    analysis_code_version VARCHAR,
    created_at      VARCHAR
);

CREATE TABLE IF NOT EXISTS optimization_records (
    record_id       VARCHAR PRIMARY KEY,
    study_id        VARCHAR NOT NULL,
    component_id    VARCHAR NOT NULL,
    iteration       INTEGER NOT NULL,
    parameters      VARCHAR,   -- json
    objective_name  VARCHAR,
    objective_value DOUBLE,
    best_so_far     DOUBLE,
    is_best         BOOLEAN,
    run_id          VARCHAR,
    status          VARCHAR,
    created_at      VARCHAR
);
"""

# Column order per table (DuckDB positional INSERT). Must match _SCHEMA above.
_COLS = {
    "components": ["component_id", "type", "name", "library", "source_url", "function",
                   "platform", "materials", "geometry", "expected_metrics", "notes",
                   "created_at"],
    "simulation_runs": ["run_id", "component_id", "tool", "tool_version", "is_synthetic",
                        "wl_min_um", "wl_max_um", "n_wavelengths", "boundary_conditions",
                        "mesh_summary", "settings", "geometry_snapshot", "task_id",
                        "status", "flex_credit_cost", "created_at"],
    "raw_artifacts": ["artifact_id", "run_id", "kind", "path", "format",
                      "checksum_sha256", "size_bytes", "created_at"],
    "analysis_results": ["result_id", "run_id", "component_id", "metric_name",
                         "metric_value", "units", "wavelength_um", "definition",
                         "analysis_code_version", "created_at"],
    "optimization_records": ["record_id", "study_id", "component_id", "iteration",
                             "parameters", "objective_name", "objective_value",
                             "best_so_far", "is_best", "run_id", "status", "created_at"],
}

_JSON_FIELDS = {
    "components": {"materials", "geometry", "expected_metrics"},
    "simulation_runs": {"settings", "geometry_snapshot"},
    "optimization_records": {"parameters"},
}

_JSON_DECODE_COLS = {"materials", "geometry", "expected_metrics", "settings",
                     "geometry_snapshot", "parameters"}


def _encode_value(table: str, key: str, v: Any) -> Any:
    if key in _JSON_FIELDS.get(table, set()):
        return json.dumps(v)
    if isinstance(v, datetime):
        return v.isoformat()
    if hasattr(v, "value"):  # Enum
        return v.value
    return v


class PicStore:
    """Facade over the DuckDB metadata DB, the Parquet spectra store, and the
    Zarr field-array store."""

    def __init__(self, root: str | Path = "data/pipeline", read_only: bool = False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "pic.duckdb"
        self.array_dir = self.root / "arrays"      # Parquet spectra
        self.field_dir = self.root / "fields"      # Zarr N-D arrays
        self.array_dir.mkdir(exist_ok=True)
        self.field_dir.mkdir(exist_ok=True)
        self.read_only = read_only
        if read_only:
            self.con = duckdb.connect(str(self.db_path), read_only=True)
        else:
            self.con = duckdb.connect(str(self.db_path))
            self.con.execute(_SCHEMA)

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "PicStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------- generic upsert ----------
    def _upsert(self, table: str, model) -> None:
        data = model.model_dump()
        cols = _COLS[table]
        values = [_encode_value(table, c, data.get(c)) for c in cols]
        placeholders = ", ".join(["?"] * len(cols))
        self.con.execute(
            f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({placeholders})",
            values,
        )

    # ---------- writers ----------
    def add_component(self, c: Component) -> str:
        self._upsert("components", c)
        return c.component_id

    def add_run(self, r: SimulationRun) -> str:
        self._upsert("simulation_runs", r)
        return r.run_id

    def add_artifact(self, a: RawArtifact) -> str:
        self._upsert("raw_artifacts", a)
        return a.artifact_id

    def add_result(self, r: AnalysisResult) -> str:
        self._upsert("analysis_results", r)
        return r.result_id

    def add_optimization_record(self, o: OptimizationRecord) -> str:
        self._upsert("optimization_records", o)
        return o.record_id

    # ---------- tabular array payloads (Parquet) ----------
    def write_spectrum(self, run_id: str, df: pd.DataFrame) -> Path:
        """Persist a spectrum / S-parameter table as Parquet; return the path."""
        path = self.array_dir / f"{run_id}.parquet"
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path,
                       compression="snappy")
        return path

    def read_spectrum(self, run_id: str) -> pd.DataFrame:
        path = self.array_dir / f"{run_id}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No spectrum stored for run {run_id}")
        return pq.read_table(path).to_pandas()

    # ---------- N-D array payloads (Zarr) ----------
    def write_field(self, run_id: str, array: np.ndarray,
                    coords: Optional[dict[str, Any]] = None,
                    attrs: Optional[dict[str, Any]] = None) -> Path:
        path = self.field_dir / f"{run_id}.zarr"
        z = zarr.open(
            str(path), mode="w", shape=array.shape,
            chunks=tuple(min(s, 128) for s in array.shape), dtype=array.dtype,
        )
        z[:] = array
        z.attrs["coords"] = json.loads(json.dumps(coords or {}, default=list))
        z.attrs.update(attrs or {})
        return path

    def read_field(self, run_id: str) -> tuple[np.ndarray, dict]:
        path = self.field_dir / f"{run_id}.zarr"
        if not path.exists():
            raise FileNotFoundError(f"No field array stored for run {run_id}")
        z = zarr.open(str(path), mode="r")
        return z[:], dict(z.attrs)

    # ---------- query helpers ----------
    def _df(self, sql: str, params: Optional[list] = None) -> pd.DataFrame:
        df = self.con.execute(sql, params or []).df()
        for col in df.columns:
            if col in _JSON_DECODE_COLS:
                df[col] = df[col].apply(lambda s: json.loads(s) if isinstance(s, str) else s)
        return df

    def components(self) -> pd.DataFrame:
        return self._df("SELECT * FROM components ORDER BY created_at")

    def runs_for_component(self, component_id: str) -> pd.DataFrame:
        """Fetch all simulation runs for a component."""
        return self._df(
            "SELECT * FROM simulation_runs WHERE component_id = ? ORDER BY created_at",
            [component_id],
        )

    def metrics(self, component_id: Optional[str] = None,
                metric_name: Optional[str] = None) -> pd.DataFrame:
        """Retrieve performance metrics joined with run + component metadata."""
        sql = """
            SELECT ar.result_id, ar.metric_name, ar.metric_value, ar.units,
                   ar.wavelength_um, ar.definition,
                   c.component_id, c.type, c.name AS component_name,
                   sr.run_id, sr.tool, sr.is_synthetic
            FROM analysis_results ar
            JOIN simulation_runs sr ON sr.run_id = ar.run_id
            JOIN components c ON c.component_id = ar.component_id
            WHERE 1=1
        """
        params: list[Any] = []
        if component_id:
            sql += " AND c.component_id = ?"
            params.append(component_id)
        if metric_name:
            sql += " AND ar.metric_name = ?"
            params.append(metric_name)
        sql += " ORDER BY c.component_id, ar.metric_name"
        return self._df(sql, params)

    def filter_runs_by_geometry(self, key: str, min_val=None, max_val=None) -> pd.DataFrame:
        """Filter runs by a geometry parameter stored inside the JSON snapshot."""
        val_expr = "CAST(json_extract(geometry_snapshot, '$.' || ?) AS DOUBLE)"
        sql = (
            f"SELECT run_id, component_id, {val_expr} AS geom_value, "
            "geometry_snapshot, is_synthetic, created_at "
            "FROM simulation_runs "
            f"WHERE {val_expr} IS NOT NULL"
        )
        params: list[Any] = [key, key]
        if min_val is not None:
            sql += f" AND {val_expr} >= ?"
            params += [key, min_val]
        if max_val is not None:
            sql += f" AND {val_expr} <= ?"
            params += [key, max_val]
        sql += " ORDER BY geom_value"
        return self._df(sql, params)

    def compare_variants(self, component_type: str, metric_name: str) -> pd.DataFrame:
        """Compare a single metric across all variants (runs) of a component type.

        One row per run, with the run's own geometry_snapshot — so this works
        whether variants are modeled as separate components or as multiple runs
        of one component.
        """
        sql = """
            SELECT c.component_id, c.type, c.name AS component_name,
                   sr.run_id, sr.geometry_snapshot,
                   ar.metric_value, ar.units, sr.is_synthetic
            FROM analysis_results ar
            JOIN components c ON c.component_id = ar.component_id
            JOIN simulation_runs sr ON sr.run_id = ar.run_id
            WHERE c.type = ? AND ar.metric_name = ?
            ORDER BY ar.metric_value
        """
        return self._df(sql, [component_type, metric_name])

    def compare_runs(self, component_id: str, metric_name: str) -> pd.DataFrame:
        """Per-run metric + geometry for one component (variant comparison)."""
        sql = """
            SELECT sr.run_id, sr.geometry_snapshot, ar.metric_value, ar.units
            FROM analysis_results ar
            JOIN simulation_runs sr ON sr.run_id = ar.run_id
            WHERE ar.component_id = ? AND ar.metric_name = ?
            ORDER BY ar.metric_value
        """
        return self._df(sql, [component_id, metric_name])

    def optimization_history(self, study_id: str) -> pd.DataFrame:
        return self._df(
            "SELECT * FROM optimization_records WHERE study_id = ? ORDER BY iteration",
            [study_id],
        )

    def summary_table(self) -> pd.DataFrame:
        """Per (component, metric) aggregate across that component's runs."""
        sql = """
            SELECT c.name AS component, c.type, sr.tool,
                   CASE WHEN sr.is_synthetic THEN 'synthetic' ELSE 'real' END AS data_source,
                   ar.metric_name,
                   ROUND(AVG(ar.metric_value), 4) AS mean_value,
                   ROUND(MIN(ar.metric_value), 4) AS best_value,
                   COUNT(*) AS n_runs, ar.units
            FROM analysis_results ar
            JOIN components c ON c.component_id = ar.component_id
            JOIN simulation_runs sr ON sr.run_id = ar.run_id
            GROUP BY c.name, c.type, sr.tool, data_source, ar.metric_name, ar.units
            ORDER BY c.type, c.name, ar.metric_name
        """
        return self._df(sql)
