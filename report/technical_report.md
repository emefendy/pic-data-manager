# Technical Report — PIC Development Data Pipeline (real Tidy3D data)

**Package:** `picpipe` (v0.2) · **Challenge:** QuEra Photonics Data Engineer take-home, Option 1

---

## 1. Overview

`picpipe` is a reproducible data pipeline for photonic-integrated-circuit (PIC)
development. It ingests this project's **real** Tidy3D FDTD sweep results, stores
them in a structured, queryable format with full provenance, computes photonic
performance metrics, and supports variant comparison and a geometry-optimization
loop. There is **no synthetic data** — every record traces back to a real FDTD
run produced by `scripts/sweep_yjunction.py` / `scripts/sweep_eulerbend.py`.

One command runs the whole thing:

```bash
python -m picpipe.cli demo
```

It ingests 53 real designs (31 Y-junctions, 22 Euler bends), computes metrics,
runs the optimization, and writes four figures + a summary table to
`data/pipeline/figures/`.

## 2. System architecture

| Layer | Module | Responsibility |
|---|---|---|
| Component registry | `picpipe/models.py`, `picpipe/components.py` | Pydantic models; the two reference components |
| Simulation adapter | `picpipe/ingest.py` | Real Tidy3D spectra/HDF5 → normalized records |
| Data store | `picpipe/store.py` | DuckDB (metadata + query) + Parquet (spectra) + Zarr (fields) |
| Analysis | `picpipe/analysis.py` | Insertion loss, extinction ratio, 3 dB bandwidth + sanity checks |
| Optimization | `picpipe/optimize.py` | Parameter-sweep optimization over the real grid |
| Demo / CLI | `picpipe/demo.py`, `picpipe/cli.py` | Single-command end-to-end run + queries |
| Real solver workflow | `scripts/sweep_yjunction.py`, `scripts/sweep_eulerbend.py`, `scripts/extract_sim_data.py` | The executable Tidy3D FDTD runs that produced the data |

The **ingestion adapter is the single seam** between a raw FDTD result and its
stored metrics: `ingest_tidy3d_design` reads a `pic_data.db` design row + its
`data/spectra/<id>.json` transmission curve, and `ingest_spectrum` ensures the
component, creates a `SimulationRun` with provenance, writes the spectrum to
Parquet, registers a checksummed `RawArtifact` (including a pointer to the raw
FDTD HDF5), and computes + stores the metrics. No manual copy/paste step.

## 3. Data model and storage

### 3.1 Schema

Five Pydantic entities (`models.py`), covering the challenge's minimum schema:

- **Component** — `component_id`, `type`, `library`/`source_url`, `function`,
  `platform`, `materials`, `geometry`, `expected_metrics`.
- **SimulationRun** — `run_id`, `component_id`, `tool`, `tool_version`,
  `is_synthetic`, `wl_min/max_um`, `n_wavelengths`, `boundary_conditions`,
  `mesh_summary`, `settings`, `geometry_snapshot`, `task_id`, `created_at`.
- **RawArtifact** — `path` + `format` + `checksum_sha256` for each Parquet
  spectrum and each raw FDTD HDF5.
- **AnalysisResult** — `metric_name`, `metric_value`, `units`, `definition`,
  `analysis_code_version`.
- **OptimizationRecord** — `study_id`, `iteration`, `parameters`,
  `objective_value`, `best_so_far`, `is_best`, `run_id`.

Content-hashed IDs make ingestion idempotent (re-running the demo does not
duplicate rows). Each device family is one Component; each swept design is a
`SimulationRun` of it, so "all simulations for a component" returns all 31
Y-junction runs and "compare variants" compares them by geometry.

### 3.2 Storage — DuckDB + Parquet + Zarr

- **DuckDB** (`data/pipeline/pic.duckdb`) holds all metadata, provenance, and
  metrics, and is the SQL query engine (incl. JSON extraction of geometry).
- **Parquet** (`data/pipeline/arrays/`) holds the 1-D transmission spectra, one
  file per run.
- **Zarr** is available for N-D field-monitor cubes (unused by the 1-D spectra
  here, but wired in for field maps).

This consolidates the project's two pipelines under one storage philosophy:
`picpipe` (simulation) and `screening` (the ScreeningDatasetV2 lab data) both use
DuckDB + Parquet + Zarr.

### 3.3 Query layer

`PicStore` exposes the required operations: `runs_for_component()`,
`metrics(component_id, metric_name)`, `compare_variants(type, metric)` /
`compare_runs(component_id, metric)`, and `filter_runs_by_geometry(key, min, max)`
via DuckDB JSON extraction. All are reachable from the CLI (`query`, `filter`,
`summary`).

## 4. Simulation integration (Tidy3D)

The real FDTD workflow is `scripts/sweep_yjunction.py` and
`scripts/sweep_eulerbend.py`: they build `td.Simulation` objects, submit
`web.Job`s, download HDF5, and extract transmission spectra to
`data/spectra/*.json` (91 HDF5 files on disk). `picpipe.ingest.ingest_tidy3d_hdf5`
also parses a raw `SimulationData` directly. The pipeline stores each result with
component identity, geometry parameters (`geometry_snapshot`), tool + version,
boundary conditions, timestamps, and a checksum to the raw artifact — i.e. enough
provenance to reproduce or compare runs.

**Assumptions / limitations (for a production version):**
- Geometry is stored as a free-form JSON snapshot; a production schema would add
  per-family typed geometry tables and a canonical units convention.
- Provenance should additionally capture mesh statistics, convergence/divergence
  flags, the solver/PDK version, and an environment lockfile per run.
- The 3 dB bandwidth here reads as the full simulated band (100 nm) because the
  swept spectra are flat to within 3 dB over 1.50–1.60 µm; a wider sweep is
  needed to resolve a true roll-off. (Note: the legacy `simulated_bandwidth_ghz`
  column in `pic_data.db` is unreliable due to a unit bug in the original
  `extract_sim_data.py`; `picpipe` recomputes bandwidth correctly in nm.)

## 5. Analysis results (real data)

From `python -m picpipe.cli demo`:

| Component | n runs | mean insertion loss | best |
|---|---|---|---|
| Y-junction (1×2 splitter) | 31 | 0.465 dB | 0.155 dB |
| Euler bend | 22 | 0.027 dB | 0.005 dB |

Metric definitions follow Appendix A (insertion loss = −10·log10 T, in dB;
bandwidth = width within 3 dB of minimum loss, in nm). The demo produces four
figures: insertion-loss vs wavelength (cross-component), Y-junction variant
comparison (mean loss vs `l_junction`), per-component loss distribution, and the
optimization-progress curve.

## 6. Stretch — optimization

`optimize_over_real_sweep` performs a reproducible gradient-free parameter sweep
over the **real** Y-junction grid: it walks `l_junction` in order, treats each
design's measured mean insertion loss as the objective, tracks best-so-far, and
stores every candidate as an `OptimizationRecord` linked to its `SimulationRun`.
On the real data it selects **l_junction = 2.0 µm** (mean loss **0.155 dB**), a
**0.54 dB improvement** over the baseline design. Constraint: candidates come from
the precomputed FDTD grid (no new solver calls); a finer search would pass a
Tidy3D-backed evaluator.

## 7. Design tradeoffs

- **DuckDB + Parquet vs. the project's legacy SQLite** — chose DuckDB so metadata
  and columnar spectra live under one analytical engine, consistent with the
  `screening` pipeline; the legacy `pic_data.db` remains for the dashboards.
- **Free-form JSON geometry** — extensible across component types without schema
  migrations; validated at the Pydantic layer.
- **One Component per family, many runs** — makes "all sims for a component" and
  variant comparison natural.
- **Content-hashed IDs** — idempotent ingestion over append-only history.

## 8. Production next steps

1. Object storage (S3) for Parquet/HDF5 with keys in `RawArtifact.path`; Postgres
   if multi-writer concurrency is needed (`PicStore` is the only thing to change).
2. Richer provenance (mesh stats, convergence flags, PDK/solver version, lockfile).
3. Typed geometry per component family + a units registry; fix and backfill the
   legacy bandwidth column.
4. CI with these tests + golden-file metric regressions.
5. Bayesian/adjoint optimization with parallel real-FDTD evaluation and cost tracking.
6. Extend the model to fabrication + measured test data for sim-vs-measured yield.

## 9. Validation

13 pytest tests (`tests/test_picpipe.py`) cover the analysis metrics (incl. the
nm bandwidth convention), model validators, store ingest round-trip + all query
ops, idempotency, geometry filtering, the optimization loop, and a real-dataset
integration test that ingests `pic_data.db`.
```bash
pytest tests/test_picpipe.py -q
```
