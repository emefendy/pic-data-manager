"""Characterization analysis for ScreeningDatasetV2.

Three metric families, each with an explicit unit and a documented definition:

1. Static fringe (from ``static_vdc_sweep``): the combined-MZI transmission vs
   static bias. Metrics: V_null, PD_min/PD_max, extinction/visibility, and a
   V_pi estimate from the spacing of adjacent fringe extrema.
2. Small-signal Bode (from ``fra_smallsignal``): DC gain, -3 dB bandwidth, and
   the phase at the -3 dB point.
3. Per-capture step features (from each ``captures/*.npz``): trace-averaged
   baseline/top levels, peak-to-peak swing, 10-90% rise time, and overshoot.

All functions are pure (array in, numbers out) so they are easy to unit-test;
the pipeline wraps them into AnalysisResult records.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

CODE_VERSION = "screening-analysis-0.1.0"


# --------------------------------------------------------------------------- #
# 1. Static fringe
# --------------------------------------------------------------------------- #
def static_fringe_metrics(v_dc: np.ndarray, pd_mean: np.ndarray) -> dict[str, tuple]:
    """Return {metric_name: (value, units, definition)} for the static sweep."""
    v = np.asarray(v_dc, dtype=float)
    p = np.asarray(pd_mean, dtype=float)

    i_min = int(np.argmin(p))
    i_max = int(np.argmax(p))
    pmin, pmax = float(p[i_min]), float(p[i_max])
    v_null = float(v[i_min])

    # visibility / extinction of the fringe
    visibility = (pmax - pmin) / (pmax + pmin) if (pmax + pmin) > 0 else float("nan")
    extinction_db = 10.0 * np.log10(pmax / pmin) if pmin > 0 else float("nan")

    # V_pi estimate: half the bias spacing between adjacent like-extrema.
    # Find maxima and minima; the null->peak spacing is V_pi.
    span = pmax - pmin
    peaks, _ = find_peaks(p, prominence=0.25 * span)
    nulls, _ = find_peaks(-p, prominence=0.25 * span)
    extrema = np.sort(np.concatenate([peaks, nulls]).astype(int))
    if len(extrema) >= 2:
        # mean bias distance between consecutive extrema = V_pi (null<->peak)
        v_pi = float(np.mean(np.abs(np.diff(v[extrema]))))
    else:
        v_pi = float("nan")

    out = {
        "static_V_null": (v_null, "V", "Static bias of minimum PD output (fringe null)"),
        "static_PD_min": (pmin * 1e3, "mV", "Minimum averaged PD output over the sweep"),
        "static_PD_max": (pmax * 1e3, "mV", "Maximum averaged PD output over the sweep"),
        "static_visibility": (visibility, "ratio", "(PD_max-PD_min)/(PD_max+PD_min)"),
        "static_extinction": (extinction_db, "dB", "10*log10(PD_max/PD_min)"),
    }
    if np.isfinite(v_pi):
        out["static_V_pi_est"] = (
            v_pi, "V", "Mean bias spacing between adjacent fringe extrema (null<->peak)")
    return out


# --------------------------------------------------------------------------- #
# 2. Small-signal Bode / FRA
# --------------------------------------------------------------------------- #
def bode_metrics(freq_hz: np.ndarray, h_mag_db: np.ndarray,
                 h_phase_rad: np.ndarray) -> dict[str, tuple]:
    f = np.asarray(freq_hz, dtype=float)
    mag = np.asarray(h_mag_db, dtype=float)
    ph = np.asarray(h_phase_rad, dtype=float)
    order = np.argsort(f)
    f, mag, ph = f[order], mag[order], ph[order]

    # DC gain: average of the lowest-frequency decade
    lowband = f <= (f.min() * 10)
    dc_gain_db = float(np.mean(mag[lowband])) if lowband.any() else float(mag[0])

    # -3 dB bandwidth: first crossing below (dc_gain - 3 dB)
    target = dc_gain_db - 3.0
    f3db = float("nan")
    phase_at_f3db = float("nan")
    below = np.where(mag <= target)[0]
    if len(below) > 0:
        k = int(below[0])
        if k > 0:  # linear interpolate in log-f for the crossing
            f1, f2 = np.log10(f[k - 1]), np.log10(f[k])
            m1, m2 = mag[k - 1], mag[k]
            frac = (target - m1) / (m2 - m1) if m2 != m1 else 0.0
            f3db = float(10 ** (f1 + frac * (f2 - f1)))
        else:
            f3db = float(f[k])
        phase_at_f3db = float(np.interp(np.log10(f3db), np.log10(f), ph))

    return {
        "fra_dc_gain": (dc_gain_db, "dB", "Mean |H| over the lowest frequency decade"),
        "fra_bandwidth_3db": (f3db / 1e6 if np.isfinite(f3db) else float("nan"), "MHz",
                              "Frequency where |H| falls 3 dB below the DC gain"),
        "fra_phase_at_3db": (np.degrees(phase_at_f3db) if np.isfinite(phase_at_f3db)
                             else float("nan"), "deg", "Phase of H at the -3 dB frequency"),
    }


# --------------------------------------------------------------------------- #
# 3. Per-capture step features
# --------------------------------------------------------------------------- #
def capture_step_features(time_s: np.ndarray, ch1: np.ndarray) -> dict[str, tuple]:
    """Trace-averaged step features from one capture.

    ``ch1`` is (n_traces, N) or (N,). Levels are taken from the lower/upper
    percentiles of the averaged trace (robust to noise and ringing).
    """
    t = np.asarray(time_s, dtype=float)
    y = np.asarray(ch1, dtype=float)
    mean_trace = y.mean(axis=0) if y.ndim == 2 else y

    baseline = float(np.percentile(mean_trace, 5))
    top = float(np.percentile(mean_trace, 95))
    swing = top - baseline

    # 10-90% rise time on the averaged trace (largest rising transition)
    lo, hi = baseline + 0.1 * swing, baseline + 0.9 * swing
    rise_s = float("nan")
    try:
        above_hi = np.where(mean_trace >= hi)[0]
        above_lo = np.where(mean_trace >= lo)[0]
        if len(above_hi) and len(above_lo):
            i_hi, i_lo = int(above_hi[0]), int(above_lo[0])
            if i_hi > i_lo:
                rise_s = float(t[i_hi] - t[i_lo])
    except Exception:
        pass

    overshoot = (float(mean_trace.max()) - top) / swing * 100.0 if swing > 0 else float("nan")
    # per-trace noise: mean std across traces (if a cube)
    noise_mv = float(np.mean(y.std(axis=0))) * 1e3 if y.ndim == 2 else float("nan")

    return {
        "cap_baseline_mV": (baseline * 1e3, "mV", "5th-percentile level of averaged trace"),
        "cap_top_mV": (top * 1e3, "mV", "95th-percentile level of averaged trace"),
        "cap_swing_mV": (swing * 1e3, "mV", "Peak-to-peak swing (top-baseline)"),
        "cap_rise_time_ns": (rise_s * 1e9 if np.isfinite(rise_s) else float("nan"), "ns",
                             "10-90% rise time of the averaged trace"),
        "cap_overshoot_pct": (overshoot, "%", "Overshoot above the top level, % of swing"),
        "cap_trace_noise_mV": (noise_mv, "mV", "Mean per-sample std across traces"),
    }
