"""Ingestion adapters: ScreeningDatasetV2 raw files -> linked pipeline records.

One adapter per data type. Each ensures the device exists, creates a Measurement
with full provenance (instrument banner, acquisition params, timestamp), persists
the array payload (Parquet table or Zarr cube) as a checksummed RawArtifact, and
computes + stores the relevant AnalysisResults. There is no manual step between a
raw file and its stored metrics.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from screening import analysis
from screening.models import (
    AnalysisResult,
    ArtifactFormat,
    Device,
    LabLog,
    Measurement,
    MeasurementKind,
    RawArtifact,
    new_id,
)
from screening.store import ScreeningStore

DEVICE_ID = "cascaded_mzi_v2"


def _scalar(v: Any) -> Any:
    """Convert a 0-d numpy array / numpy scalar to a Python scalar."""
    if isinstance(v, np.ndarray) and v.ndim == 0:
        v = v.item()
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    return v


def _store_results(store: ScreeningStore, measurement_id: str, device_id: str,
                   metrics: dict[str, tuple], group: str) -> None:
    for name, (value, units, definition) in metrics.items():
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            continue  # skip undefined metrics rather than store NaN
        store.add_result(AnalysisResult(
            result_id=new_id("res", measurement_id, name),
            measurement_id=measurement_id, device_id=device_id,
            metric_name=name, metric_value=float(value), units=units,
            group=group, definition=definition,
            analysis_code_version=analysis.CODE_VERSION,
        ))


# --------------------------------------------------------------------------- #
def ingest_device_card(store: ScreeningStore, card_path: str) -> str:
    """Parse device_card.md into the Device record (best-effort, keeps raw text)."""
    text = Path(card_path).read_text() if Path(card_path).exists() else ""
    known, unknowns = [], []
    mode = None
    for line in text.splitlines():
        s = line.strip()
        low = s.lower()
        if low.startswith("**what we know"):
            mode = "known"; continue
        if low.startswith("**what we don't know") or low.startswith("**what we dont know"):
            mode = "unknown"; continue
        if s.startswith("**") and mode:  # next bold heading ends the list
            mode = None
        if s.startswith("- ") and mode == "known":
            known.append(s[2:].strip())
        elif s.startswith("- ") and mode == "unknown":
            unknowns.append(s[2:].strip())

    device = Device(
        device_id=DEVICE_ID,
        name="Cascaded MZI modulator (ScreeningDatasetV2)",
        type="cascaded_mzi_modulator",
        description="Two-stage cascaded Mach-Zehnder modulator, PD + internal TIA readout.",
        pd_bandwidth_hz=150e6, ac_coupling_cutoff_hz=10.0, n_stages=2,
        known_facts=known, unknowns=unknowns, source_card=text,
    )
    store.add_device(device)
    return device.device_id


def ingest_static_sweep(store: ScreeningStore, npz_path: str,
                        device_id: str = DEVICE_ID) -> str:
    d = np.load(npz_path, allow_pickle=True)
    v_dc = np.asarray(d["V_DC"], dtype=float)
    pd_mean = np.asarray(d["PD_mean"], dtype=float)
    pd_std = np.asarray(d["PD_std"], dtype=float)
    n_traces = int(_scalar(d["n_traces"])) if "n_traces" in d.files else None

    mid = new_id("meas", device_id, "static_sweep")
    store.add_measurement(Measurement(
        measurement_id=mid, device_id=device_id, kind=MeasurementKind.STATIC_SWEEP,
        session_banner=str(_scalar(d.get("session_banner", ""))),
        params={"V_DC_min": float(v_dc.min()), "V_DC_max": float(v_dc.max())},
        n_points=int(v_dc.size), n_traces=n_traces,
        note=str(_scalar(d.get("note", ""))),
    ))

    df = pd.DataFrame({"V_DC_volts": v_dc, "PD_mean_mV": pd_mean * 1e3,
                       "PD_std_mV": pd_std * 1e3,
                       "error_on_mean_mV": pd_std * 1e3 / np.sqrt(n_traces or 1)})
    path = store.write_table(mid, df)
    store.add_artifact(RawArtifact(
        measurement_id=mid, artifact_id=new_id("art", mid, "static_table"),
        kind="static_table", path=str(path), format=ArtifactFormat.PARQUET,
        checksum_sha256=RawArtifact.sha256_of(str(path)),
        size_bytes=Path(path).stat().st_size))

    _store_results(store, mid, device_id,
                   analysis.static_fringe_metrics(v_dc, pd_mean), group="metric")
    return mid


def ingest_fra(store: ScreeningStore, npz_path: str, device_id: str = DEVICE_ID) -> str:
    d = np.load(npz_path, allow_pickle=True)
    freq = np.asarray(d["freq_hz"], dtype=float)
    mag = np.asarray(d["H_mag_db"], dtype=float)
    phase = np.asarray(d["H_phase_rad"], dtype=float)

    mid = new_id("meas", device_id, "fra")
    store.add_measurement(Measurement(
        measurement_id=mid, device_id=device_id, kind=MeasurementKind.FRA,
        session_banner=str(_scalar(d.get("session_banner", ""))),
        params={"drive_vpp": float(_scalar(d.get("drive_vpp", np.nan))),
                "V_DC": float(_scalar(d.get("V_DC", np.nan))),
                "freq_min_hz": float(freq.min()), "freq_max_hz": float(freq.max())},
        n_points=int(freq.size), note=str(_scalar(d.get("note", ""))),
    ))

    df = pd.DataFrame({"freq_hz": freq, "H_mag_db": mag, "H_phase_rad": phase,
                       "H_phase_deg": np.degrees(phase)})
    path = store.write_table(mid, df)
    store.add_artifact(RawArtifact(
        measurement_id=mid, artifact_id=new_id("art", mid, "fra_table"),
        kind="fra_table", path=str(path), format=ArtifactFormat.PARQUET,
        checksum_sha256=RawArtifact.sha256_of(str(path)),
        size_bytes=Path(path).stat().st_size))

    _store_results(store, mid, device_id,
                   analysis.bode_metrics(freq, mag, phase), group="metric")
    return mid


def ingest_capture(store: ScreeningStore, npz_path: str,
                   device_id: str = DEVICE_ID, store_full_traces: bool = True) -> str:
    d = np.load(npz_path, allow_pickle=True)
    tag = str(_scalar(d["tag"]))
    time = np.asarray(d["time"], dtype="float32")
    ch1 = np.asarray(d["ch1"], dtype="float32")
    ch3 = np.asarray(d["ch3"], dtype="float32")
    rep_hz = float(_scalar(d["rep_hz"]))
    edge_s = float(_scalar(d["edge_s"]))

    mid = new_id("meas", device_id, tag)
    params = {
        "tag": tag, "rep_hz": rep_hz, "edge_s": edge_s, "edge_ns": edge_s * 1e9,
        "V_DC": float(_scalar(d.get("V_DC", np.nan))),
        "V_pp": float(_scalar(d.get("V_pp", np.nan))),
        "vlbl": str(_scalar(d.get("vlbl", ""))),
        "period_s": float(_scalar(d.get("period_s", np.nan))),
        "sample_rate_hz": float(_scalar(d.get("sample_rate_hz", np.nan))),
    }
    store.add_measurement(Measurement(
        measurement_id=mid, device_id=device_id, kind=MeasurementKind.CAPTURE, tag=tag,
        session_banner=str(_scalar(d.get("session_banner", ""))), params=params,
        n_points=int(time.size), n_traces=int(ch1.shape[0]) if ch1.ndim == 2 else 1,
    ))

    # waveform cube -> Zarr. Always store the trace-averaged ch1/ch3 plus the
    # per-sample std (so a ±σ band is available without the full cube). The full
    # (n_traces, N) cubes are optional, to keep the committed store light.
    arrays = {
        "time": time,
        "ch1_mean": ch1.mean(axis=0) if ch1.ndim == 2 else ch1,
        "ch1_std": ch1.std(axis=0) if ch1.ndim == 2 else np.zeros_like(ch1),
        "ch3_mean": ch3.mean(axis=0) if ch3.ndim == 2 else ch3,
    }
    if store_full_traces:
        arrays["ch1"] = ch1
        arrays["ch3"] = ch3
    path = store.write_waveform(tag, arrays, attrs=params)
    store.add_artifact(RawArtifact(
        measurement_id=mid, artifact_id=new_id("art", mid, "waveform"),
        kind="waveform_cube", path=str(path), format=ArtifactFormat.ZARR,
        checksum_sha256=None, size_bytes=int(ch1.nbytes + ch3.nbytes + time.nbytes)))

    _store_results(store, mid, device_id,
                   analysis.capture_step_features(time, ch1), group="feature")
    return mid


def ingest_captures(store: ScreeningStore, captures_dir: str,
                    device_id: str = DEVICE_ID, limit: Optional[int] = None,
                    store_full_traces: bool = True) -> list[str]:
    files = sorted(glob.glob(str(Path(captures_dir) / "*.npz")))
    if limit:
        files = files[:limit]
    return [ingest_capture(store, f, device_id, store_full_traces) for f in files]


# Match capture-tag families mentioned in prose, e.g. "rep_300kHz_*_sym10"
# or "rep300kHz_e10ns_sym1". Normalized to "rep<rate>kHz" form below.
_TAG_RE = re.compile(r"rep_?(\d+)\s*kHz")


def ingest_lab_logs(store: ScreeningStore, logs_dir: str,
                    device_id: str = DEVICE_ID) -> list[str]:
    ids = []
    for p in sorted(glob.glob(str(Path(logs_dir) / "*.md"))):
        text = Path(p).read_text()
        first = next((ln for ln in text.splitlines() if ln.strip()), "")
        title = first.lstrip("# ").strip()
        date_m = re.search(r"(\d{4}-\d{2}-\d{2})", Path(p).name)
        refs = sorted(set(f"rep{m.group(1)}kHz" for m in _TAG_RE.finditer(text)))
        log = LabLog(
            log_id=new_id("log", device_id, Path(p).name),
            device_id=device_id, date=date_m.group(1) if date_m else None,
            title=title, body=text, referenced_tags=refs, source_path=str(p),
        )
        store.add_lab_log(log)
        ids.append(log.log_id)
    return ids
