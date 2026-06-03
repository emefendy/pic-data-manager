"""picpipe — a PIC development data pipeline over REAL Tidy3D simulation data.

Ingests this project's real FDTD sweep outputs (data/spectra/*.json produced by
scripts/sweep_yjunction.py and scripts/sweep_eulerbend.py, indexed in
pic_data.db) into a structured, queryable store and computes photonic
performance metrics for analysis, comparison, and optimization.

Modules
-------
models      : Pydantic data model (Component, SimulationRun, RawArtifact,
              AnalysisResult, OptimizationRecord) with provenance.
store       : DuckDB (metadata/query) + Parquet (spectra) + Zarr (fields).
analysis    : Insertion loss, extinction ratio, 3 dB bandwidth, sanity checks.
ingest      : Adapters that map real Tidy3D spectra/HDF5 into pipeline records.
components  : The two reference components (Y-junction, Euler bend).
optimize    : Geometry optimization over the real sweep (stretch challenge).
sim         : Executable real Tidy3D workflow (refactored from the notebooks).
"""

from picpipe.models import (
    AnalysisResult,
    Component,
    OptimizationRecord,
    RawArtifact,
    SimulationRun,
)
from picpipe.store import PicStore

__version__ = "0.2.0"

__all__ = [
    "Component", "SimulationRun", "RawArtifact", "AnalysisResult",
    "OptimizationRecord", "PicStore", "__version__",
]
