<p align="center">
  <img src="docs/dashboard_tidy3d.png" width="24%" />
  <img src="docs/dashboard_labdata.png" width="24%" />
  <img src="docs/dashboard_queries.png" width="24%" />
  <img src="docs/dashboard_optimization.png" width="24%" />
</p>

A reproducible data platform for photonic-integrated-circuit (PIC) development,
covering two **real** datasets end to end:

- **Tidy3D simulations** — 53 real FDTD designs (31 Y-junctions, 22 Euler bends)
  swept on the Tidy3D cloud, with full provenance.
- **Real lab data** — a cascaded Mach–Zehnder modulator characterization
  (static V_DC sweep, small-signal Bode, 100 step-pulse captures, operator lab logs).

Both flow through the same storage stack — **DuckDB** (metadata, provenance,
metrics, SQL queries) + **Parquet** (tabular arrays) + **Zarr** (N-D arrays) —
into an interactive Streamlit dashboard with one-click + natural-language querying,
RandomForest + SHAP modeling on the lab data, and a **closed-loop Bayesian
optimizer**: a Gaussian process seeded with the existing sweep proposes new
(w7, h_bend, l_junction) combinations, each one runs as a real Tidy3D job, and
the results feed straight back into the store.

![Pipeline flow chart](pipeline_flowchart.png)

## Repository structure

```
pic-data-manager/
├── tidy3d_pipeline/            # simulation pipeline: models · store · ingest · analysis · optimize · cli
├── reallabdata_pipeline/       # lab-data pipeline:  models · store · ingest · analysis · cli
├── scripts/
│   ├── sweep_yjunction.py      # run real FDTD sweeps on the Tidy3D cloud (costs FlexCredits)
│   ├── sweep_eulerbend.py      # each combo also runs a circular-bend baseline (2 jobs/combo)
│   ├── optimize_yjunction.py   # closed-loop Bayesian optimization (GP + EI → real FDTD jobs)
│   └── build_manifest.py       # designs.json → data/tidy3d_raw/manifest.parquet
├── data/
│   ├── tidy3d_raw/             # RAW sim results: spectra (JSON) + manifest (Parquet)   [committed]
│   ├── reallabdata_raw/        # RAW lab dataset: captures (NPZ), sweeps, lab logs      [committed]
│   ├── tidy3d_processed/       # store: tidy3d.duckdb + arrays/*.parquet + fields/*.zarr [generated]
│   └── reallabdata_processed/  # store: reallabdata.duckdb + tables + waveforms/*.zarr   [generated]
├── notebooks/                  # interactive Tidy3D prototypes + raw HDF5 outputs
├── docs/                       # dashboard screenshots (used above)
├── app_real.py                 # the Streamlit dashboard
├── dashboard_data.py           # headless data/analytics layer (no Streamlit; unit-testable)
├── tests/                      # pytest suites for both pipelines + metadata validation
├── technical_report.pdf        # architecture, schema, ingestion flow, results, tradeoffs
└── pipeline_flowchart.png      # the flow chart above
```

## Setup

```bash
git clone git@github.com:emefendy/pic-data-manager.git
cd pic-data-manager
pip install -r requirements.txt
```

**Dependencies** (`requirements.txt`):

- `pandas`, `numpy`, `scipy`, `pydantic` — core
- `duckdb`, `pyarrow`, `zarr`, `h5py` — storage
- `scikit-learn`, `shap`, `statsmodels` — modeling / attribution
- `streamlit`, `plotly` — dashboard
- `anthropic` — natural-language → SQL queries (optional; needs `ANTHROPIC_API_KEY`)
- `tidy3d`, `photonforge`, `gdstk` — only needed to run **new** FDTD sweeps
  (requires a Flexcompute account + FlexCredits)
- `jupyterlab` — notebooks

### Natural-language queries (API key)

The dashboard's natural-language → SQL boxes use the Anthropic API. To enable
them, create `.streamlit/secrets.toml` (gitignored) with the key below:

```toml
ANTHROPIC_API_KEY = "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"
```

Everything else in the dashboard works without it.

## Run the demo

One command — ingestion, analysis, and the dashboard:

```bash
streamlit run app_real.py    # builds both stores automatically on first launch
```
