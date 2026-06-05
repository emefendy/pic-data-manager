from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

CODE_VERSION = "screening-analysis-0.1.0"


def static_fringe_metrics(v_dc: np.ndarray, pd_mean: np.ndarray) -> dict[str, tuple]:
    v = np.asarray(v_dc, dtype=float)
    p = np.asarray(pd_mean, dtype=float)

    i_min = int(np.argmin(p))
    i_max = int(np.argmax(p))
    pmin, pmax = float(p[i_min]), float(p[i_max])
    v_null = float(v[i_min])

    visibility = (pmax - pmin) / (pmax + pmin) if (pmax + pmin) > 0 else float("nan")
    extinction_db = 10.0 * np.log10(pmax / pmin) if pmin > 0 else float("nan")

    span = pmax - pmin
    peaks, _ = find_peaks(p, prominence=0.25 * span)
    nulls, _ = find_peaks(-p, prominence=0.25 * span)
    extrema = np.sort(np.concatenate([peaks, nulls]).astype(int))
    if len(extrema) >= 2:
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


def bode_metrics(freq_hz: np.ndarray, h_mag_db: np.ndarray,
                 h_phase_rad: np.ndarray) -> dict[str, tuple]:
    f = np.asarray(freq_hz, dtype=float)
    mag = np.asarray(h_mag_db, dtype=float)
    ph = np.asarray(h_phase_rad, dtype=float)
    order = np.argsort(f)
    f, mag, ph = f[order], mag[order], ph[order]

    lowband = f <= (f.min() * 10)
    dc_gain_db = float(np.mean(mag[lowband])) if lowband.any() else float(mag[0])

    target = dc_gain_db - 3.0
    f3db = float("nan")
    phase_at_f3db = float("nan")
    below = np.where(mag <= target)[0]
    if len(below) > 0:
        k = int(below[0])
        if k > 0:
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


def capture_step_features(time_s: np.ndarray, ch1: np.ndarray) -> dict[str, tuple]:
    t = np.asarray(time_s, dtype=float)
    y = np.asarray(ch1, dtype=float)
    mean_trace = y.mean(axis=0) if y.ndim == 2 else y

    baseline = float(np.percentile(mean_trace, 5))
    top = float(np.percentile(mean_trace, 95))
    swing = top - baseline

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
