"""screening - a data pipeline for ScreeningDatasetV2.

Ingests real cascaded Mach-Zehnder modulator characterization data (a static
V_DC sweep, a small-signal Bode/FRA, 100 step-pulse captures, and operator lab
logs) into a structured, queryable store and computes device-characterization
metrics for analysis and visualization.

Storage stack (same philosophy as the picpipe PIC pipeline):
* DuckDB  - device/measurement metadata, provenance, features, metrics, lab logs;
            the SQL query engine.
* Parquet - tabular array payloads (static sweep, FRA/Bode tables, feature table).
* Zarr    - N-D capture waveforms (ch1/ch3 trace cubes, shape (n_traces, N)).

Entry points:
    python -m screening.cli demo      # ingest + analyze + visualize, end to end
    python -m screening.cli query ...
"""

from screening.models import (
    AnalysisResult,
    Device,
    LabLog,
    Measurement,
    RawArtifact,
)
from screening.store import ScreeningStore

__version__ = "0.1.0"

__all__ = [
    "Device",
    "Measurement",
    "RawArtifact",
    "AnalysisResult",
    "LabLog",
    "ScreeningStore",
    "__version__",
]
