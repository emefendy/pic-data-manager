from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("::".join(map(str, parts)).encode()).hexdigest()[:10]
    return f"{prefix}_{digest}"


class MeasurementKind(str, Enum):
    STATIC_SWEEP = "static_sweep"
    FRA = "fra"
    CAPTURE = "capture"


class ArtifactFormat(str, Enum):
    NPZ = "npz"
    PARQUET = "parquet"
    ZARR = "zarr"
    MARKDOWN = "markdown"
    JSON = "json"


class Device(BaseModel):

    device_id: str
    name: str
    type: str = "cascaded_mzi_modulator"
    description: str = ""
    pd_bandwidth_hz: Optional[float] = None
    ac_coupling_cutoff_hz: Optional[float] = None
    n_stages: int = 2
    known_facts: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    source_card: str = Field("", description="Raw device_card.md text")
    created_at: datetime = Field(default_factory=utcnow)

    @field_validator("device_id", "name")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not str(v).strip():
            raise ValueError("must be a non-empty string")
        return v


class Measurement(BaseModel):

    measurement_id: str
    device_id: str
    kind: MeasurementKind
    tag: Optional[str] = Field(None, description="capture tag, e.g. rep30kHz_e10ns_sym1")
    instrument: str = "Moku"
    session_banner: str = ""
    stamp: Optional[str] = Field(None, description="acquisition timestamp string from the file")
    params: dict[str, Any] = Field(default_factory=dict)
    n_points: Optional[int] = None
    n_traces: Optional[int] = None
    note: str = ""
    created_at: datetime = Field(default_factory=utcnow)


class RawArtifact(BaseModel):
    measurement_id: str
    artifact_id: str
    kind: str = Field(..., description="e.g. static_table, fra_table, waveform_cube")
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

    result_id: str
    measurement_id: str
    device_id: str
    metric_name: str
    metric_value: float
    units: str
    group: str = Field("metric", description="'metric' (device-level) or 'feature' (per-capture)")
    definition: str = ""
    analysis_code_version: str = "screening-0.1.0"
    created_at: datetime = Field(default_factory=utcnow)

    @field_validator("metric_value")
    @classmethod
    def _finite(cls, v: float) -> float:
        if v is None or math.isnan(v) or math.isinf(v):
            raise ValueError("metric_value must be finite")
        return v


class LabLog(BaseModel):

    log_id: str
    device_id: str
    date: Optional[str] = None
    title: str = ""
    body: str = ""
    referenced_tags: list[str] = Field(
        default_factory=list, description="capture tags mentioned in the note"
    )
    source_path: str = ""
    created_at: datetime = Field(default_factory=utcnow)
