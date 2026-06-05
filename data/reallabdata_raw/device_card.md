# Device card

**Device.** Cascaded Mach–Zehnder interferometer (MZI) modulator with two electrically-independent
phase shifters, one per stage. Combined optical output is read by a photodiode (PD) with internal
transimpedance amplifier, ~150 MHz electrical bandwidth, AC-coupled.

**Static behavior.** The combined transmission is *not* a single cos² fringe. The two stages each
contribute their own phase modulation, and the static V_DC sweep visible in `static_vdc_sweep.npz`
reflects their combination. Inferring whether the stages combine multiplicatively (cascaded MZI
physics) or additively (independent transmission paths) is part of Q1.

**Drive chain.**
- Drive waveform stored as a 16-bit LUT inside the AWG.
- AWG output: 50 Ω source-impedance, ±5 V rail (DAC limit).
- Cable run: ~1 m of coax, no in-line attenuator.
- Device pad: high-impedance AC input.
- Static bias is supplied by an external DC source through a bias-T and combined with the AC drive
  at the device pad.

**Readout.**
- PD output → internal TIA → Moku ADC.
- AC-coupled; low-frequency cutoff ~10 Hz.
- PD electrical bandwidth ~150 MHz.

**What we know is true on this device** (so candidates aren't sent on a wild goose chase):
- Optical path is single-polarization (a polarization controller is upstream).
- The two phase shifters are nominally identical in design but their fabricated V_π values may differ.
- The bench has been running stably for ~8 weeks; V_null drifts ~±200 mV on a ~30 min AC thermal cycle.

**What we don't know** (so candidates can speculate):
- Whether stage-to-stage optical loss is symmetric.
- Whether the two stages share a common ground reference (this matters for Q2's impedance question).
