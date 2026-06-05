# ScreeningDatasetV2

Cascaded Mach–Zehnder modulator characterization data for the QuEra Photonics Data Engineer
Intern screening problem v2.

## Files

| File | Purpose |
|---|---|
| `static_vdc_sweep.npz` | Static V_DC sweep of PD_mean over ±5 V; no AC drive. Use for **Q1**. |
| `fra_smallsignal.npz` | Small-signal Bode plot (mag + phase) at quadrature. Use for **Q1.4** and **Q2.3**. |
| `captures/` | 100 step-pulse captures at 5 rep rates × 4 edge times × 5 drive-amplitude variants. Use for **Q2**. |
| `lab_log_examples/` | 5 short markdown operator notes. Use for **Q3** (the RAG seam). |
| `manifest_v1_captures.json` | Per-capture metadata for the 100 step-pulse files in `captures/`. |
| `device_card.md` | What's known and unknown about the device. |
| `README.md` | This file. |

## Schema

### `static_vdc_sweep.npz`
- `V_DC` — static bias voltages applied via the DC source, in V. Shape `(N,)`, float32.
- `PD_mean` — PD output voltage at each V_DC, averaged across `n_traces`. In V. Shape `(N,)`.
- `PD_std` — per-trace standard deviation (not the std of the mean). In V. Shape `(N,)`.
- `n_traces` — int, number of averaged traces per point.
- `session_banner` — instrument session metadata recorded at run open.
- `note` — short freeform note.

### `fra_smallsignal.npz`
- `freq_hz` — frequency vector. Shape `(M,)`, float32. Spans roughly 100 Hz to 150 MHz.
- `H_mag_db` — magnitude of the device transfer function H(f) in dB. Shape `(M,)`.
- `H_phase_rad` — phase of H(f) in radians. Shape `(M,)`.
- `drive_vpp` — small-signal AWG drive amplitude (around the static V_DC bias).
- `V_DC` — static bias at which the FRA was acquired (chosen near quadrature of the fringe).
- `session_banner`, `note` — context.

### `captures/*.npz` (from v1 of this problem)
Each capture is a single sweep cell at one (rep rate, edge time, drive-amplitude variant). Inside:
- `time` — time axis (s), shape `(N,)`.
- `ch1` — averaged PD readout (V), shape `(n_traces, N)` or `(N,)`.
- `ch3` — drive trigger / monitor (V), shape `(n_traces, N)` or `(N,)`.
- Metadata: `rep_hz`, `edge_s`, `period_s`, `V_pp`, `V_DC`, `vlbl`, `tag`, etc.

Drive-amplitude variant labels in `vlbl`:
- `sym1` — symmetric ±0.5 V (1 V_pp)
- `sym5` — symmetric ±2.5 V (5 V_pp)
- `sym10` — symmetric ±5 V (10 V_pp)
- `asymHi` — 0 V to +5 V (5 V_pp, biased high)
- `asymLo` — −5 V to 0 V (5 V_pp, biased low)

All step-pulse captures share the same static V_DC and the same device as the static sweep above.

## Notes

- PD noise on the static sweep is approximately Gaussian, σ ≈ 0.5 mV per trace (so σ/√n ≈ 0.09 mV
  on `PD_mean` with `n_traces=32`). `PD_std` reports the per-trace value, not σ/√n.
- The FRA was acquired at a static bias near quadrature so the device responds linearly to small
  drive. Reading the Bode plot does not require any knowledge of which stage dominates — that's
  Q1's question.
- The operator notes in `lab_log_examples/` are real-style entries inspired by the kinds of events
  that happen on this bench. They are paired with `manifest.json` and a captures directory in
  production; here only the markdown is shipped.
