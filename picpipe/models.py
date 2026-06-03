"""Data model for the PIC development data pipeline.

These Pydantic models are the single source of truth for the pipeline schema.
They map directly onto the "Minimum Data Schema Requirements" of the challenge:

    Component        -> a uniquely identified PIC design (variant).
    SimulationRun    -> one electromagnetic simulation, with enough provenance
                        to reproduce or compare it.
    RawArtifact      -> a pointer (path/key + checksum) to a large raw file,
                        so big arrays are NOT forced into relational tables.
    AnalysisResult   -> one computed metric (name, value, units, code version).
    OptimizationRecord -> one iteration of a geometry optimization loop.

Design notes
------------
* IDs are explicit strings so records are stable and portable across stores.
* ``geometry`` and ``settings`` are free-form dicts: every component type has
  different geometry parameters, and forcing a fixed column set would not be
  extensible. They are stored as JSON in DuckDB (see ``store.py``).
* Timestamps are timezone-aware UTC.
* Validators enforce the few invariants that catch the most common ingestion
  bugs (empty IDs, NaN metric values, unknown status strings).
"""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


def utcnow() -> datetime:
    """Timezone-aware current UTC time (used as the default timestamp)."""
    return datetime.now(timezone.utc)


def new_id(prefix: str, *parts: str) -> str:
    """Deterministic short id built from ``parts`` (stable across re-ingestion).

    Using a content hash means re-ingesting the same logical record yields the
    same id, which keeps the pipeline idempotent instead of creating duplicates.
    """
    digest = hashlib.sha1("::".join(map(str, parts)).encode()).hexdigest()[:10]
    return f"{prefix}_{digest}"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"


class ArtifactFormat(str, Enum):
    HDF5 = "hdf5"
    PARQUET = "parquet"
    ZARR = "zarr"  # chunked N-D arrays (field/monitor cubes)
    NPZ = "npz"
    JSON = "json"
    CSV = "csv"
    TOUCHSTONE = "touchstone"  # .sNp S-parameter files


class Component(BaseModel):
    """A PIC component / design variant.

    Required fields (challenge schema): component_id, type, library/source,
    geometry parameters, material/platform assumptions.
    """

    component_id: str
    type: str = Field(..., description="e.g. y_junction, euler_bend, directional_coupler")
    name: str
    library: str = Field(..., description="Source library / reference, e.g. 'Tidy3D examples'")
    source_url: Optional[str] = None
    function: Optional[str] = Field(None, description="What the component does")
    platform: str = Field("SOI-220nm", description="Material/process platform")
    materials: dict[str, str] = Field(default_factory=dict, description="role -> material")
    geometry: dict[str, Any] = Field(default_factory=dict, description="Geometry parameters")
    expected_metrics: dict[str, Any] = Field(
        default_factory=dict, description="Datasheet/target performance metrics"
    )
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)

    @field_validator("component_id", "type", "name", "library")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not str(v).strip():
            raise ValueError("must be a non-empty string")
        return v


class SimulationRun(BaseModel):
    """One simulation run with provenance sufficient to reproduce/compare it.

    Required fields: run_id, component_id, tool, version, settings,
    wavelength/frequency range, boundary conditions, mesh/settings summary.
    """

    run_id: str
    component_id: str
    tool: str = Field(..., description="e.g. tidy3d, meep")
    tool_version: str = "unknown"
    is_synthetic: bool = Field(
        False, description="True for clearly-labeled synthetic data (no real solver run)"
    )
    wl_min_um: float = Field(..., description="Minimum wavelength in microns")
    wl_max_um: float = Field(..., description="Maximum wavelength in microns")
    n_wavelengths: int = Field(..., ge=2)
    boundary_conditions: str = "PML"
    mesh_summary: str = Field("", description="e.g. 'auto grid, min_steps_per_wvl=20'")
    settings: dict[str, Any] = Field(default_factory=dict)
    geometry_snapshot: dict[str, Any] = Field(
        default_factory=dict, description="Geometry params actually used for this run"
    )
    task_id: Optional[str] = Field(None, description="Solver/cloud task id, if any")
    status: RunStatus = RunStatus.SUCCESS
    flex_credit_cost: Optional[float] = None
    created_at: datetime = Field(default_factory=utcnow)

    @field_validator("wl_max_um")
    @classmethod
    def _wl_order(cls, v: float, info) -> float:
        wmin = info.data.get("wl_min_um")
        if wmin is not None and v <= wmin:
            raise ValueError(f"wl_max_um ({v}) must be > wl_min_um ({wmin})")
        return v


class RawArtifact(BaseModel):
    """A reference to a large raw file kept outside the relational tables.

    Required fields: artifact_id, run_id, path or object key, format,
    checksum or version tag.
    """

    artifact_id: str
    run_id: str
    kind: str = Field(..., description="e.g. spectrum, field_monitor, s_parameter")
    path: str
    format: ArtifactFormat
    checksum_sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    created_at: datetime = Field(default_factory=utcnow)

    @staticmethod
    def sha256_of(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()


class AnalysisResult(BaseModel):
    """One computed performance metric.

    Required fields: result_id, run_id, metric_name, metric_value, units,
    analysis_code_version.
    """

    result_id: str
    run_id: str
    component_id: str
    metric_name: str = Field(..., description="e.g. insertion_loss_mean, bandwidth_3db")
    metric_value: float
    units: str
    wavelength_um: Optional[float] = Field(
        None, description="Wavelength the metric is reported at, if applicable"
    )
    definition: str = Field("", description="Human-readable definition / convention")
    analysis_code_version: str = "picpipe-0.1.0"
    created_at: datetime = Field(default_factory=utcnow)

    @field_validator("metric_value")
    @classmethod
    def _finite(cls, v: float) -> float:
        if v is None or math.isnan(v) or math.isinf(v):
            raise ValueError("metric_value must be a finite number")
        return v


class OptimizationRecord(BaseModel):
    """One iteration of a geometry optimization loop (stretch challenge).

    Required fields: iteration, parameter set, objective value, best-so-far, status.
    """

    record_id: str
    study_id: str = Field(..., description="Groups iterations of one optimization study")
    component_id: str
    iteration: int = Field(..., ge=0)
    parameters: dict[str, Any]
    objective_name: str
    objective_value: float
    best_so_far: float
    is_best: bool = False
    run_id: Optional[str] = Field(None, description="SimulationRun produced for this candidate")
    status: str = "evaluated"
    created_at: datetime = Field(default_factory=utcnow)
