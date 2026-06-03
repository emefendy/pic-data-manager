# Real-data dashboard (`app_real.py`)

A Streamlit dashboard that shows **only real data** — no synthetic/seeded rows —
organized into two clearly separated domains, each with the full analytical
treatment (overview, records, parameter/process analysis, anomaly insights,
predictive model).

## Run

```bash
cd /Users/emilie/pic-data-manager
pip install -r requirements.txt
streamlit run app_real.py
```

Pick a dataset at the top:

**🔷 Tidy3D PIC Simulations** — your real FDTD designs from `pic_data.db`
(only `designer in ('sweep','simulation_import')`, i.e. the Y-junction + Euler-bend
sweeps; synthetic seed designs are excluded). Tabs:

- Overview — designs per type, insertion-loss distribution
- Design Library — filterable real-design table with geometry
- Spectra — transmission/insertion-loss vs wavelength from `data/spectra/*.json`
- Parameter Analysis — geometry ↔ loss correlations + scatter (e.g. `l_junction`
  correlates ≈ −0.99 with loss)
- Anomaly Insights — IsolationForest flags designs with unusual geometry→loss behavior
- Predictive Model — cross-validated RandomForest predicting loss from geometry (R² ≈ 0.99)

**🔬 ScreeningDatasetV2 — Lab Data** — the cascaded-MZI characterization, served
from the `screening/` pipeline's DuckDB+Parquet+Zarr store. On first use it builds
the store from the raw dataset (defaults to `~/Downloads/ScreeningDatasetV2_candidate`,
or set `SCREENING_DATASET`; there's also a "Build store" button). Tabs:

- Overview — device metrics (−3 dB BW, V_null, V_π, extinction) + device card
- Captures — filter the 100 captures by rep rate / drive variant
- Characterization — static fringe + small-signal Bode
- Waveform — per-capture PD trace (mean ±1σ) read from Zarr
- Parameter Analysis — drive params ↔ step-response correlations + scatter
- Anomaly Insights — IsolationForest on capture features (cross-references the
  thermal-drift lab log)
- Predictive Model — predict PD swing from drive parameters (R² ≈ 0.99)
- Lab Logs — the operator notes (RAG seam)

## Notes

- This dashboard is **additive and non-destructive**: your original `app.py` and
  `pic_data.db` are left untouched. `app_real.py` simply *excludes* synthetic data
  rather than deleting it.
- The pure data + analytics layer lives in `dashboard_data.py` (no Streamlit), so
  it can be unit-tested headless.
- If you want me to go further and **fully purge** the synthetic generator
  (`scripts/seed_data.py` + the seeded rows in `pic_data.db`) and make this the
  default `app.py`, that's a one-step follow-up — just say so.
