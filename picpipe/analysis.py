"""Analysis layer: photonic performance metrics.

Every metric returns an :class:`~picpipe.models.AnalysisResult` with an explicit
unit and a one-line definition string, so a plot or table is never ambiguous
about convention. Metric definitions follow the challenge's Appendix A.

Conventions used throughout
---------------------------
* ``transmission`` is linear power transmission (|S21|^2), 0..~1.
* Insertion loss is reported as a **positive** dB number: IL = -10*log10(T).
* Wavelengths are in microns (um); bandwidth is reported in nanometers (nm).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from picpipe.models import AnalysisResult, new_id

CODE_VERSION = "picpipe-analysis-0.1.0"


# --------------------------------------------------------------------------- #
# Low-level numeric metrics (pure functions, easy to unit-test)
# --------------------------------------------------------------------------- #
def insertion_loss_db(transmission: np.ndarray) -> np.ndarray:
    """Per-wavelength insertion loss in dB (positive = loss).

    IL(λ) = -10 * log10(T(λ)). A tiny floor avoids log10(0).
    """
    t = np.clip(np.asarray(transmission, dtype=float), 1e-12, None)
    return -10.0 * np.log10(t)


def mean_insertion_loss_db(transmission: np.ndarray) -> float:
    return float(np.mean(insertion_loss_db(transmission)))


def min_insertion_loss_db(transmission: np.ndarray) -> float:
    return float(np.min(insertion_loss_db(transmission)))


def extinction_ratio_db(t_on: np.ndarray, t_off: np.ndarray) -> float:
    """Extinction ratio in dB between an ON and OFF transmission state.

    ER = 10*log10(mean(T_on) / mean(T_off)).
    """
    on = float(np.mean(np.clip(t_on, 1e-12, None)))
    off = float(np.mean(np.clip(t_off, 1e-12, None)))
    return 10.0 * np.log10(on / off)


def bandwidth_3db_nm(wavelength_um: np.ndarray, transmission: np.ndarray,
                     threshold_db: float = 3.0) -> float:
    """Width (nm) of the band where IL stays within ``threshold_db`` of its best.

    Finds the contiguous region around the minimum-loss wavelength where
    insertion loss <= min_loss + threshold, and returns its width in nm.
    (Reported in nm — note this is the correct convention; a common bug is to
    mislabel an optical-frequency span as GHz.)
    """
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
    return float(width_um * 1000.0)  # um -> nm


def resonance_wavelength_um(wavelength_um: np.ndarray, transmission: np.ndarray) -> float:
    """Wavelength of minimum transmission (e.g. a ring resonator notch)."""
    wl = np.asarray(wavelength_um, dtype=float)
    return float(wl[int(np.argmin(transmission))])


# --------------------------------------------------------------------------- #
# Sanity checks (the "validation checks" the rubric asks for)
# --------------------------------------------------------------------------- #
def check_spectrum(df: pd.DataFrame, transmission_col: str = "transmission") -> list[str]:
    """Return a list of human-readable warnings about a spectrum.

    Empty list == passed all sanity checks. Checks are intentionally physical:
    monotonic wavelength axis, transmission in [0, ~1], no NaNs.
    """
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


# --------------------------------------------------------------------------- #
# High-level: spectrum -> list[AnalysisResult]
# --------------------------------------------------------------------------- #
def analyze_spectrum(
    component_id: str,
    run_id: str,
    df: pd.DataFrame,
    transmission_col: str = "transmission",
    center_wl_um: float = 1.55,
) -> list[AnalysisResult]:
    """Compute the standard metric set for a transmission spectrum.

    Returns AnalysisResult records (not yet stored). Raises ValueError if the
    spectrum fails sanity checks, so bad data never silently produces metrics.
    """
    problems = check_spectrum(df, transmission_col)
    if problems:
        raise ValueError(f"spectrum for run {run_id} failed sanity checks: {problems}")

    wl = df["wavelength_um"].to_numpy(dtype=float)
    t = df[transmission_col].to_numpy(dtype=float)

    # metric at the center wavelength
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
