from __future__ import annotations

import numpy as np
import pandas as pd

from tidy3d_pipeline.models import AnalysisResult, new_id

CODE_VERSION = "tidy3d_pipeline-analysis-0.1.0"


def insertion_loss_db(transmission: np.ndarray) -> np.ndarray:
    t = np.clip(np.asarray(transmission, dtype=float), 1e-12, None)
    return -10.0 * np.log10(t)


def mean_insertion_loss_db(transmission: np.ndarray) -> float:
    return float(np.mean(insertion_loss_db(transmission)))


def min_insertion_loss_db(transmission: np.ndarray) -> float:
    return float(np.min(insertion_loss_db(transmission)))


def extinction_ratio_db(t_on: np.ndarray, t_off: np.ndarray) -> float:
    on = float(np.mean(np.clip(t_on, 1e-12, None)))
    off = float(np.mean(np.clip(t_off, 1e-12, None)))
    return 10.0 * np.log10(on / off)


def bandwidth_3db_nm(wavelength_um: np.ndarray, transmission: np.ndarray,
                     threshold_db: float = 3.0) -> float:
    wl = np.asarray(wavelength_um, dtype=float)
    il = insertion_loss_db(transmission)
    best_idx = int(np.argmin(il))
    cutoff = il[best_idx] + threshold_db

    lo = best_idx
    while lo > 0 and il[lo - 1] <= cutoff:
        lo -= 1
    hi = best_idx
    while hi < len(il) - 1 and il[hi + 1] <= cutoff:
        hi += 1
    width_um = wl[hi] - wl[lo]
    return float(width_um * 1000.0)


def resonance_wavelength_um(wavelength_um: np.ndarray, transmission: np.ndarray) -> float:
    wl = np.asarray(wavelength_um, dtype=float)
    return float(wl[int(np.argmin(transmission))])


def check_spectrum(df: pd.DataFrame, transmission_col: str = "transmission") -> list[str]:
    warnings: list[str] = []
    if "wavelength_um" not in df.columns:
        warnings.append("missing 'wavelength_um' column")
        return warnings
    if transmission_col not in df.columns:
        warnings.append(f"missing '{transmission_col}' column")
        return warnings

    wl = df["wavelength_um"].to_numpy(dtype=float)
    t = df[transmission_col].to_numpy(dtype=float)

    if np.isnan(wl).any() or np.isnan(t).any():
        warnings.append("contains NaN values")
    if not (np.all(np.diff(wl) > 0) or np.all(np.diff(wl) < 0)):
        warnings.append("wavelength axis is not monotonic")
    if (t < -1e-9).any():
        warnings.append("transmission has negative values")
    if (t > 1.05).any():
        warnings.append("transmission exceeds 1.05 (non-physical for a passive device)")
    return warnings


def analyze_spectrum(
    component_id: str,
    run_id: str,
    df: pd.DataFrame,
    transmission_col: str = "transmission",
    center_wl_um: float = 1.55,
) -> list[AnalysisResult]:
    problems = check_spectrum(df, transmission_col)
    if problems:
        raise ValueError(f"spectrum for run {run_id} failed sanity checks: {problems}")

    wl = df["wavelength_um"].to_numpy(dtype=float)
    t = df[transmission_col].to_numpy(dtype=float)

    ci = int(np.argmin(np.abs(wl - center_wl_um)))

    specs = [
        ("insertion_loss_mean", mean_insertion_loss_db(t), "dB", None,
         "Mean of -10*log10(T) over the simulated band"),
        ("insertion_loss_min", min_insertion_loss_db(t), "dB", None,
         "Best-case -10*log10(T) over the band"),
        ("insertion_loss_at_center", float(insertion_loss_db(t)[ci]), "dB", float(wl[ci]),
         "-10*log10(T) at the center wavelength"),
        ("bandwidth_3db", bandwidth_3db_nm(wl, t), "nm", None,
         "Width of the band within 3 dB of minimum insertion loss"),
    ]

    results = []
    for name, value, units, at_wl, definition in specs:
        results.append(
            AnalysisResult(
                result_id=new_id("res", run_id, name),
                run_id=run_id,
                component_id=component_id,
                metric_name=name,
                metric_value=value,
                units=units,
                wavelength_um=at_wl,
                definition=definition,
                analysis_code_version=CODE_VERSION,
            )
        )
    return results
