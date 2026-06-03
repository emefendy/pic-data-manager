"""Ingestion adapters — REAL Tidy3D outputs -> linked pipeline records.

The pipeline ingests this project's real FDTD results:

* ``ingest_spectrum`` — the single choke point: ensures the component exists,
  creates a SimulationRun (full provenance), persists the spectrum as Parquet,
  registers a checksummed RawArtifact, and computes + stores AnalysisResults.
  No manual step between a raw spectrum and its stored metrics.
* ``ingest_tidy3d_design`` — maps one swept design (a pic_data.db row + its
  data/spectra/<id>.json transmission curve) onto ``ingest_spectrum``.
* ``ingest_real_dataset`` — ingests every real design in pic_data.db.
* ``ingest_tidy3d_hdf5`` — parses a raw Tidy3D ``SimulationData`` HDF5 directly
  (requires tidy3d); used when starting from raw solver output.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from picpipe.analysis import analyze_spectrum
from picpipe.components import COMPONENTS, TRANSMISSION_KEY
from picpipe.models import Component, RawArtifact, SimulationRun, new_id
from picpipe.store import PicStore

REAL_DESIGNERS = ("sweep", "simulation_import")


def ingest_spectrum(
    store: PicStore,
    component: Component,
    spectrum: pd.DataFrame,
    *,
    tool: str = "tidy3d",
    tool_version: str = "unknown",
    is_synthetic: bool = False,
    geometry_snapshot: Optional[dict[str, Any]] = None,
    settings: Optional[dict[str, Any]] = None,
    task_id: Optional[str] = None,
    flex_credit_cost: Optional[float] = None,
    mesh_summary: str = "",
    boundary_conditions: str = "PML",
    transmission_col: str = "transmission",
    center_wl_um: float = 1.55,
    run_suffix: str = "",
    raw_artifact_path: Optional[str] = None,
) -> str:
    """Ingest one spectrum end-to-end. Returns the created ``run_id``."""
    if "wavelength_um" not in spectrum.columns:
        raise ValueError("spectrum must have a 'wavelength_um' column")
    store.add_component(component)

    wl = spectrum["wavelength_um"].to_numpy(dtype=float)
    geometry_snapshot = geometry_snapshot or dict(component.geometry)

    run = SimulationRun(
        run_id=new_id("run", component.component_id, tool, str(geometry_snapshot), run_suffix),
        component_id=component.component_id,
        tool=tool, tool_version=tool_version, is_synthetic=is_synthetic,
        wl_min_um=float(wl.min()), wl_max_um=float(wl.max()), n_wavelengths=int(len(wl)),
        boundary_conditions=boundary_conditions, mesh_summary=mesh_summary,
        settings=settings or {}, geometry_snapshot=geometry_snapshot,
        task_id=task_id, flex_credit_cost=flex_credit_cost,
    )
    store.add_run(run)

    # spectrum payload -> Parquet, registered as a checksummed artifact
    path = store.write_spectrum(run.run_id, spectrum)
    store.add_artifact(RawArtifact(
        artifact_id=new_id("art", run.run_id, "spectrum"), run_id=run.run_id,
        kind="spectrum", path=str(path), format="parquet",
        checksum_sha256=RawArtifact.sha256_of(str(path)),
        size_bytes=Path(path).stat().st_size))

    # optional pointer to the raw solver artifact (e.g. the FDTD HDF5)
    if raw_artifact_path and Path(raw_artifact_path).exists():
        store.add_artifact(RawArtifact(
            artifact_id=new_id("art", run.run_id, "raw"), run_id=run.run_id,
            kind="fdtd_hdf5", path=str(raw_artifact_path), format="hdf5",
            checksum_sha256=RawArtifact.sha256_of(str(raw_artifact_path)),
            size_bytes=Path(raw_artifact_path).stat().st_size))

    for r in analyze_spectrum(component.component_id, run.run_id, spectrum,
                              transmission_col=transmission_col, center_wl_um=center_wl_um):
        store.add_result(r)
    return run.run_id


def _spectrum_df(spec: dict, device_type: str) -> Optional[pd.DataFrame]:
    """Build a (wavelength_um, transmission) frame from a spectrum json."""
    wl = None
    for k in ("wavelengths_um", "wavelength_um", "ldas"):
        if k in spec:
            wl = np.asarray(spec[k], dtype=float)
            break
    if wl is None:
        return None
    tkey = TRANSMISSION_KEY.get(device_type)
    if tkey not in spec:
        # fall back to any transmission-like key
        tkey = next((k for k in spec if k.lower().startswith("t_") or k.lower() == "transmission"), None)
    if tkey is None:
        return None
    t = np.asarray(spec[tkey], dtype=float)
    n = min(len(wl), len(t))
    return pd.DataFrame({"wavelength_um": wl[:n], "transmission": t[:n]})


def ingest_tidy3d_design(store: PicStore, design_row: dict, spectra_dir: Path) -> Optional[str]:
    """Ingest one real swept design (pic_data.db row + its spectrum json)."""
    device_type = design_row.get("device_type")
    component = COMPONENTS.get(device_type)
    if component is None:
        return None  # not one of our two reference components
    design_id = design_row["design_id"]
    spec_path = Path(spectra_dir) / f"{design_id}.json"
    if not spec_path.exists():
        return None
    spec = json.loads(spec_path.read_text())
    df = _spectrum_df(spec, device_type)
    if df is None or df.empty:
        return None

    try:
        geom = json.loads(design_row.get("sim_params") or "{}")
    except Exception:
        geom = {}
    geom = {k: v for k, v in geom.items() if isinstance(v, (int, float, str))}
    geom["design_id"] = design_id

    raw = design_row.get("simulation_file")
    raw_path = str(Path(spectra_dir).parents[0] / raw) if raw else None

    return ingest_spectrum(
        store, component, df,
        tool="tidy3d", is_synthetic=False,
        geometry_snapshot=geom,
        settings={"designer": design_row.get("designer"),
                  "design_version": design_row.get("design_version")},
        mesh_summary="Tidy3D auto grid (see sweep script)",
        boundary_conditions="PML all sides",
        run_suffix=design_id,
        raw_artifact_path=raw_path if raw_path and Path(raw_path).exists() else None,
    )


def ingest_real_dataset(store: PicStore, db_path: str = "pic_data.db",
                        spectra_dir: str = "data/spectra") -> list[str]:
    """Ingest every real Tidy3D design from pic_data.db that has a spectrum."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" * len(REAL_DESIGNERS))
        rows = con.execute(
            f"SELECT * FROM designs WHERE designer IN ({placeholders})", REAL_DESIGNERS
        ).fetchall()
    finally:
        con.close()
    run_ids = []
    for row in rows:
        rid = ingest_tidy3d_design(store, dict(row), Path(spectra_dir))
        if rid:
            run_ids.append(rid)
    return run_ids


def ingest_tidy3d_hdf5(store: PicStore, component: Component, hdf5_path: str,
                       *, monitor_name: str = "mode", n_output_arms: int = 2,
                       center_wl_um: float = 1.55) -> str:
    """Ingest a raw Tidy3D ``SimulationData`` HDF5 file (requires tidy3d)."""
    import tidy3d as td

    sim_data = td.SimulationData.from_file(hdf5_path)
    amp = sim_data[monitor_name].amps.sel(mode_index=0, direction="+")
    freqs = amp.f.values
    wl_um = td.C_0 / freqs
    T = n_output_arms * (np.abs(amp.values) ** 2)
    order = np.argsort(wl_um)
    df = pd.DataFrame({"wavelength_um": wl_um[order], "transmission": T[order]})
    return ingest_spectrum(
        store, component, df, tool="tidy3d",
        tool_version=getattr(td, "__version__", "unknown"), is_synthetic=False,
        mesh_summary="from stored SimulationData", boundary_conditions="PML all sides",
        center_wl_um=center_wl_um, run_suffix="from_hdf5", raw_artifact_path=hdf5_path)
