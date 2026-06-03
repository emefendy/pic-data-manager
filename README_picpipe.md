# picpipe — PIC Development Data Pipeline (real Tidy3D data)

The rubric-complete pipeline for Option 1, running on this project's **real**
FDTD sweeps — no synthetic data. Ingests `data/spectra/*.json` (produced by the
real Tidy3D runs in `scripts/sweep_*.py`, indexed in `pic_data.db`) into a
DuckDB + Parquet + Zarr store, computes metrics, compares variants, and runs a
geometry optimization.

## Quick start

```bash
pip install -r requirements.txt
python -m picpipe.cli demo            # ingest 53 real designs -> figures + summary
pytest tests/test_picpipe.py -q       # 13 tests

# queries
python -m picpipe.cli summary
python -m picpipe.cli query --component y_junction
python -m picpipe.cli query --metric insertion_loss_mean
python -m picpipe.cli filter --geom l_junction --min 1.0 --max 2.0
```

Outputs: `data/pipeline/` (DuckDB + Parquet) and `data/pipeline/figures/`
(4 PNGs + `summary_table.csv`). Full write-up: `report/technical_report.md`.

## How it maps to the rubric

| Task | Where |
|---|---|
| 1 — data model | `picpipe/models.py` (5 Pydantic entities) |
| 1 — storage | `picpipe/store.py` (DuckDB + Parquet + Zarr) |
| 1 — queries | `runs_for_component`, `metrics`, `compare_variants`/`compare_runs`, `filter_runs_by_geometry` |
| 1 — tests | `tests/test_picpipe.py` (13, incl. real-data integration) |
| 2 — Tidy3D adapter | `picpipe/ingest.py` (`ingest_tidy3d_design`, `ingest_tidy3d_hdf5`) + real runs in `scripts/sweep_*.py` |
| 2 — provenance store | `SimulationRun` (tool/version/geometry/bc/timestamp) + checksummed `RawArtifact` |
| 3 — two components | `picpipe/components.py` (Y-junction, Euler bend) — real FDTD artifacts |
| 3 — comparison | `data/pipeline/figures/component_comparison.png`, `variant_comparison.png` |
| 4 — one command + report | `python -m picpipe.cli demo`, `report/technical_report.md` |
| Stretch — optimization | `picpipe/optimize.py` → `optimization_progress.png` (l_junction = 2.0 µm, −0.54 dB) |

## Notes

- **No synthetic data.** Only real FDTD designs (`designer in ('sweep',
  'simulation_import')`) are ingested.
- This supersedes the separate `picpipe` that was in the Desktop project — you can
  delete `/Users/emilie/Desktop/pic-data-manager` now.
- Fixed a unit bug in the legacy bandwidth helper (`scripts/extract_sim_data.py`
  + `sweep_*.py`): `td.C_0/λ` is Hz, so GHz = `/1e9` (was `×1e3`, giving the
  bogus `1.25e16`). `pic_data.db`'s old `simulated_bandwidth_ghz` values predate
  the fix — re-run the sweeps to backfill, or rely on picpipe's correct nm
  bandwidth metric.
- The `screening/` package (ScreeningDatasetV2 lab data) and the `app_real.py`
  dashboard share the same DuckDB+Parquet+Zarr storage philosophy.
