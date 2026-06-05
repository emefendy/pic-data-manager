# Operator log — 2026-04-23 21:08

Overnight bsssr sweep showed asymmetric fringe extrema in the last 18 captures of the run (cells
rep_300kHz_*_sym10 onwards). PD_max dropped from ~71 mV early in the night to ~58 mV by morning;
PD_min stable at 0 mV. Drive metadata unchanged. Hypothesis: room temperature rose ~2°C overnight
(HVAC issue, building-side) and drifted V_null by ~0.4 V, pushing the bright extremum off the steep
part of the fringe.

Not flagging as a bench fault — looks like the AC thermal-cycle issue noted in the device card.
Re-acquire affected captures tomorrow after morning thermal stabilization.
