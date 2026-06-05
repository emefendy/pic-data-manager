# Operator log — 2026-05-19 16:50

Noticed V_null drift of ~0.3 V over the past 3 hours despite stable temperature (logged ±0.1°C in
lab). Drive amplitude metadata correct (10 Vpp); polarization controller untouched. Tried
power-cycling the DC source — drift continued. Re-extracted V_null at 17:20, then again at 17:55 —
drift ≈ 0.08 V per 30 min, monotonic.

Suspect upstream optical power drift (laser thermal stabilization?) or PD TIA drift. Not enough
evidence yet. Setting up an overnight idle monitor (PD_mean at fixed V_DC, no drive) to characterize.
