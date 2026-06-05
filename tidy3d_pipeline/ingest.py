from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from tidy3d_pipeline.analysis import analyze_spectrum
from tidy3d_pipeline.components import COMPONENTS, TRANSMISSION_KEY
from tidy3d_pipeline.models import Component, RawArtifact, SimulationRun, new_id
from tidy3d_pipeline.store import PicStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _rel(p) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(p)


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

    path = store.write_spectrum(run.run_id, spectrum)
    store.add_artifact(RawArtifact(
        artifact_id=new_id("art", run.run_id, "spectrum"), run_id=run.run_id,
        kind="spectrum", path=_rel(path), format="parquet",
        checksum_sha256=RawArtifact.sha256_of(str(path)),
        size_bytes=Path(path).stat().st_size))

    if raw_artifact_path and Path(raw_artifact_path).exists():
        store.add_artifact(RawArtifact(
            artifact_id=new_id("art", run.run_id, "raw"), run_id=run.run_id,
            kind="fdtd_hdf5", path=_rel(raw_artifact_path), format="hdf5",
            checksum_sha256=RawArtifact.sha256_of(str(raw_artifact_path)),
            size_bytes=Path(raw_artifact_path).stat().st_size))

    for r in analyze_spectrum(component.component_id, run.run_id, spectrum,
                              transmission_col=transmission_col, center_wl_um=center_wl_um):
        store.add_result(r)
    return run.run_id


def _spectrum_df(spec: dict, device_type: str) -> Optional[pd.DataFrame]:
    wl = None
    for k in ("wavelengths_um", "wavelength_um", "ldas"):
        if k in spec:
            wl = np.asarray(spec[k], dtype=float)
            break
    if wl is None:
        return None
    tkey = TRANSMISSION_KEY.get(device_type)
    if tkey not in spec:
        tkey = next((k for k in spec if k.lower().startswith("t_") or k.lower() == "transmission"), None)
    if tkey is None:
        return None
    t = np.asarray(spec[tkey], dtype=float)
    n = min(len(wl), len(t))
    return pd.DataFrame({"wavelength_um": wl[:n], "transmission": t[:n]})


def ingest_tidy3d_design(store: PicStore, design_row: dict, spectra_dir: Path) -> Optional[str]:
    device_type = design_row.get("device_type")
    component = COMPONENTS.get(device_type)
    if component is None:
        return None
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
    project_root = Path(spectra_dir).resolve().parents[1]
    raw_path = str(project_root / raw) if raw else None

    rid = ingest_spectrum(
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

    if raw_path and Path(raw_path).exists():
        try:
            ingest_field_map_hdf5(store, rid, raw_path)
        except Exception:
            pass
    return rid


def ingest_real_dataset(store: PicStore, manifest: str = "data/tidy3d_raw/manifest.parquet",
                        spectra_dir: str = "data/tidy3d_raw") -> list[str]:
    mf = Path(manifest)
    if not mf.exists():
        raise FileNotFoundError(
            f"manifest not found: {manifest}. Run `python scripts/build_manifest.py` first.")
    df = pd.read_parquet(mf)
    run_ids = []
    for _, row in df.iterrows():
        rid = ingest_tidy3d_design(store, row.to_dict(), Path(spectra_dir))
        if rid:
            run_ids.append(rid)
    return run_ids


def ingest_field_map_hdf5(store: PicStore, run_id: str, hdf5_path: str,
                          monitor_group: str = "data/1") -> Optional[str]:
    import h5py

    p = Path(hdf5_path)
    if not p.exists():
        return None
    with h5py.File(p, "r") as h:
        if f"{monitor_group}/Ex" not in h:
            return None
        comps = [h[f"{monitor_group}/{c}/__xarray_dataarray_variable__"][()]
                 for c in ("Ex", "Ey", "Ez")]
        ex_grp = h[f"{monitor_group}/Ex"]
        x = ex_grp["x"][()] if "x" in ex_grp else None
        y = ex_grp["y"][()] if "y" in ex_grp else None
        f_hz = float(ex_grp["f"][0]) if "f" in ex_grp else None

    field = np.zeros(comps[0].shape, dtype="float64")
    for c in comps:
        field += np.abs(c) ** 2
    field = np.squeeze(field).astype("float32")

    coords: dict = {"field": "abs(E)^2"}
    if x is not None:
        coords["x_um"] = np.asarray(x, dtype=float).tolist()
    if y is not None:
        coords["y_um"] = np.asarray(y, dtype=float).tolist()
    if f_hz:
        coords["wavelength_um"] = 299792458.0 * 1e6 / f_hz

    path = store.write_field(run_id, field, coords=coords,
                             attrs={"source_hdf5": str(p), "monitor": monitor_group})
    artifact = RawArtifact(
        artifact_id=new_id("art", run_id, "field_map"), run_id=run_id,
        kind="field_monitor", path=_rel(path), format="zarr",
        checksum_sha256=None, size_bytes=int(field.nbytes))
    store.add_artifact(artifact)
    return artifact.artifact_id


def ingest_tidy3d_hdf5(store: PicStore, component: Component, hdf5_path: str,
                       *, monitor_name: str = "mode", n_output_arms: int = 2,
                       center_wl_um: float = 1.55) -> str:
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
