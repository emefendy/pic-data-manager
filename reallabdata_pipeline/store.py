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

from reallabdata_pipeline.models import (
    AnalysisResult,
    Device,
    LabLog,
    Measurement,
    RawArtifact,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS devices (
    device_id       VARCHAR PRIMARY KEY,
    name            VARCHAR NOT NULL,
    type            VARCHAR,
    description     VARCHAR,
    pd_bandwidth_hz DOUBLE,
    ac_coupling_cutoff_hz DOUBLE,
    n_stages        INTEGER,
    known_facts     VARCHAR,  -- json
    unknowns        VARCHAR,  -- json
    source_card     VARCHAR,
    created_at      VARCHAR
);

CREATE TABLE IF NOT EXISTS measurements (
    measurement_id  VARCHAR PRIMARY KEY,
    device_id       VARCHAR NOT NULL,
    kind            VARCHAR NOT NULL,
    tag             VARCHAR,
    instrument      VARCHAR,
    session_banner  VARCHAR,
    stamp           VARCHAR,
    params          VARCHAR,  -- json
    n_points        INTEGER,
    n_traces        INTEGER,
    note            VARCHAR,
    created_at      VARCHAR
);

CREATE TABLE IF NOT EXISTS raw_artifacts (
    artifact_id     VARCHAR PRIMARY KEY,
    measurement_id  VARCHAR NOT NULL,
    kind            VARCHAR,
    path            VARCHAR,
    format          VARCHAR,
    checksum_sha256 VARCHAR,
    size_bytes      BIGINT,
    created_at      VARCHAR
);

CREATE TABLE IF NOT EXISTS analysis_results (
    result_id       VARCHAR PRIMARY KEY,
    measurement_id  VARCHAR NOT NULL,
    device_id       VARCHAR NOT NULL,
    metric_name     VARCHAR NOT NULL,
    metric_value    DOUBLE NOT NULL,
    units           VARCHAR,
    "group"         VARCHAR,
    definition      VARCHAR,
    analysis_code_version VARCHAR,
    created_at      VARCHAR
);

CREATE TABLE IF NOT EXISTS lab_logs (
    log_id          VARCHAR PRIMARY KEY,
    device_id       VARCHAR NOT NULL,
    date            VARCHAR,
    title           VARCHAR,
    body            VARCHAR,
    referenced_tags VARCHAR,  -- json
    source_path     VARCHAR,
    created_at      VARCHAR
);
"""

_COLS = {
    "devices": ["device_id", "name", "type", "description", "pd_bandwidth_hz",
                "ac_coupling_cutoff_hz", "n_stages", "known_facts", "unknowns",
                "source_card", "created_at"],
    "measurements": ["measurement_id", "device_id", "kind", "tag", "instrument",
                     "session_banner", "stamp", "params", "n_points", "n_traces",
                     "note", "created_at"],
    "raw_artifacts": ["artifact_id", "measurement_id", "kind", "path", "format",
                      "checksum_sha256", "size_bytes", "created_at"],
    "analysis_results": ["result_id", "measurement_id", "device_id", "metric_name",
                         "metric_value", "units", "group", "definition",
                         "analysis_code_version", "created_at"],
    "lab_logs": ["log_id", "device_id", "date", "title", "body", "referenced_tags",
                 "source_path", "created_at"],
}

_JSON_FIELDS = {
    "devices": {"known_facts", "unknowns"},
    "measurements": {"params"},
    "lab_logs": {"referenced_tags"},
}
_JSON_DECODE_COLS = {"known_facts", "unknowns", "params", "referenced_tags"}


def _encode(table: str, key: str, v: Any) -> Any:
    if key in _JSON_FIELDS.get(table, set()):
        return json.dumps(v)
    if isinstance(v, datetime):
        return v.isoformat()
    if hasattr(v, "value"):
        return v.value
    return v


class ScreeningStore:
    def __init__(self, root: str | Path = "data/reallabdata_processed", read_only: bool = False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "reallabdata.duckdb"
        self.table_dir = self.root / "tables"
        self.wave_dir = self.root / "waveforms"
        self.table_dir.mkdir(exist_ok=True)
        self.wave_dir.mkdir(exist_ok=True)
        self.read_only = read_only
        if read_only:
            self.con = duckdb.connect(str(self.db_path), read_only=True)
        else:
            self.con = duckdb.connect(str(self.db_path))
            self.con.execute(_SCHEMA)

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "ScreeningStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _upsert(self, table: str, model) -> None:
        data = model.model_dump()
        cols = _COLS[table]
        values = [_encode(table, c, data.get(c)) for c in cols]
        quoted = ", ".join(f'"{c}"' for c in cols)
        placeholders = ", ".join(["?"] * len(cols))
        self.con.execute(
            f"INSERT OR REPLACE INTO {table} ({quoted}) VALUES ({placeholders})", values
        )

    def add_device(self, d: Device) -> str:
        self._upsert("devices", d); return d.device_id

    def add_measurement(self, m: Measurement) -> str:
        self._upsert("measurements", m); return m.measurement_id

    def add_artifact(self, a: RawArtifact) -> str:
        self._upsert("raw_artifacts", a); return a.artifact_id

    def add_result(self, r: AnalysisResult) -> str:
        self._upsert("analysis_results", r); return r.result_id

    def add_lab_log(self, l: LabLog) -> str:
        self._upsert("lab_logs", l); return l.log_id

    def write_table(self, measurement_id: str, df: pd.DataFrame) -> Path:
        path = self.table_dir / f"{measurement_id}.parquet"
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path,
                       compression="snappy")
        return path

    def read_table(self, measurement_id: str) -> pd.DataFrame:
        path = self.table_dir / f"{measurement_id}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No table stored for {measurement_id}")
        return pq.read_table(path).to_pandas()

    def write_waveform(self, tag: str, arrays: dict[str, np.ndarray],
                       attrs: Optional[dict[str, Any]] = None) -> Path:
        path = self.wave_dir / f"{tag}.zarr"
        g = zarr.open_group(str(path), mode="w")
        for name, arr in arrays.items():
            arr = np.asarray(arr)
            chunks = tuple(min(s, 256) for s in arr.shape) or (1,)
            if hasattr(g, "create_dataset"):
                g.create_dataset(name, data=arr, chunks=chunks, overwrite=True)
            else:
                z = None
                for kw in ({"chunks": chunks}, {"chunk_shape": chunks}, {}):
                    try:
                        z = g.create_array(name, shape=arr.shape, dtype=arr.dtype, **kw)
                        break
                    except TypeError:
                        continue
                if z is None:
                    z = g.create_array(name, data=arr)
                else:
                    z[...] = arr
        g.attrs.update(_jsonable(attrs or {}))
        return path

    def read_waveform(self, tag: str) -> tuple[dict[str, np.ndarray], dict]:
        path = self.wave_dir / f"{tag}.zarr"
        if not path.exists():
            raise FileNotFoundError(f"No waveform stored for {tag}")
        g = zarr.open_group(str(path), mode="r")
        try:
            names = list(g.array_keys())
        except AttributeError:
            names = [k for k, _ in g.arrays()]
        arrays = {k: g[k][:] for k in names}
        return arrays, dict(g.attrs)

    def _df(self, sql: str, params: Optional[list] = None) -> pd.DataFrame:
        df = self.con.execute(sql, params or []).df()
        for c in df.columns:
            if c in _JSON_DECODE_COLS:
                df[c] = df[c].apply(lambda s: json.loads(s) if isinstance(s, str) else s)
        return df

    def devices(self) -> pd.DataFrame:
        return self._df("SELECT * FROM devices")

    def measurements(self, kind: Optional[str] = None) -> pd.DataFrame:
        if kind:
            return self._df("SELECT * FROM measurements WHERE kind = ? ORDER BY tag", [kind])
        return self._df("SELECT * FROM measurements ORDER BY kind, tag")

    def captures(self, rep_hz=None, edge_ns=None, vlbl=None) -> pd.DataFrame:
        sql = (
            "SELECT measurement_id, tag, "
            "CAST(json_extract(params,'$.rep_hz') AS DOUBLE) AS rep_hz, "
            "CAST(json_extract(params,'$.edge_ns') AS DOUBLE) AS edge_ns, "
            "json_extract_string(params,'$.vlbl') AS vlbl, "
            "CAST(json_extract(params,'$.V_pp') AS DOUBLE) AS V_pp, "
            "CAST(json_extract(params,'$.V_DC') AS DOUBLE) AS V_DC "
            "FROM measurements WHERE kind = 'capture'"
        )
        params: list[Any] = []
        if rep_hz is not None:
            sql += " AND CAST(json_extract(params,'$.rep_hz') AS DOUBLE) = ?"; params.append(float(rep_hz))
        if edge_ns is not None:
            sql += " AND CAST(json_extract(params,'$.edge_ns') AS DOUBLE) = ?"; params.append(float(edge_ns))
        if vlbl is not None:
            sql += " AND json_extract_string(params,'$.vlbl') = ?"; params.append(vlbl)
        sql += " ORDER BY rep_hz, edge_ns, vlbl"
        return self._df(sql, params)

    def metrics(self, measurement_id: Optional[str] = None,
                group: Optional[str] = None) -> pd.DataFrame:
        sql = ('SELECT result_id, measurement_id, metric_name, metric_value, units, '
               '"group", definition FROM analysis_results WHERE 1=1')
        params: list[Any] = []
        if measurement_id:
            sql += " AND measurement_id = ?"; params.append(measurement_id)
        if group:
            sql += ' AND "group" = ?'; params.append(group)
        sql += " ORDER BY metric_name"
        return self._df(sql, params)

    def feature_table(self) -> pd.DataFrame:
        long = self._df(
            "SELECT m.tag, "
            "CAST(json_extract(m.params,'$.rep_hz') AS DOUBLE) AS rep_hz, "
            "CAST(json_extract(m.params,'$.edge_ns') AS DOUBLE) AS edge_ns, "
            "json_extract_string(m.params,'$.vlbl') AS vlbl, "
            "ar.metric_name, ar.metric_value "
            "FROM analysis_results ar JOIN measurements m ON m.measurement_id = ar.measurement_id "
            'WHERE ar."group" = \'feature\''
        )
        if long.empty:
            return long
        wide = long.pivot_table(index=["tag", "rep_hz", "edge_ns", "vlbl"],
                                columns="metric_name", values="metric_value").reset_index()
        wide.columns.name = None
        return wide

    def lab_logs(self) -> pd.DataFrame:
        return self._df("SELECT * FROM lab_logs ORDER BY date")

    def device_metrics_summary(self) -> pd.DataFrame:
        return self._df(
            'SELECT metric_name, ROUND(metric_value,5) AS value, units, definition '
            'FROM analysis_results WHERE "group" = \'metric\' ORDER BY metric_name'
        )


def _jsonable(d: dict) -> dict:
    return json.loads(json.dumps(d, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
