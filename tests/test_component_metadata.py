
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tidy3d_pipeline.components import COMPONENTS

GEOMETRY_PARAMS = {
    "y_junction": ["w1", "w7", "l_junction", "t", "h_bend"],
    "euler_bend": ["R_eff", "A", "w", "t"],
}


def _designs():
    import dashboard_data as dd
    df = dd.load_tidy3d_designs()
    if df is None or df.empty:
        pytest.skip("no design manifest available")
    return df


@pytest.mark.parametrize("device_type", list(GEOMETRY_PARAMS))
def test_swept_params_match_manifest(device_type):
    df = _designs()
    sub = df[df["device_type"] == device_type]
    if sub.empty:
        pytest.skip(f"no {device_type} designs in manifest")

    actually_swept = set()
    for p in GEOMETRY_PARAMS[device_type]:
        col = f"geom_{p}"
        if col in sub.columns:
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            if vals.nunique() > 1:
                actually_swept.add(p)

    declared = set(COMPONENTS[device_type].geometry["swept_params"])
    assert declared == actually_swept, (
        f"{device_type}: metadata declares swept_params={sorted(declared)} but the "
        f"manifest actually varies {sorted(actually_swept)} — update "
        f"tidy3d_pipeline/components.py (and the components table) to match.")


@pytest.mark.parametrize("device_type", list(GEOMETRY_PARAMS))
def test_fixed_params_are_actually_fixed(device_type):
    df = _designs()
    sub = df[df["device_type"] == device_type]
    if sub.empty:
        pytest.skip(f"no {device_type} designs in manifest")

    geom = COMPONENTS[device_type].geometry
    for key, expected in geom.get("fixed_params", {}).items():
        p = key.replace("_um", "").split("_")[-1]
        col = f"geom_{p}"
        if col not in sub.columns:
            continue
        vals = pd.to_numeric(sub[col], errors="coerce").dropna()
        assert vals.nunique() == 1, f"{device_type}: {p} declared fixed but varies"
        assert float(vals.iloc[0]) == pytest.approx(float(expected)), (
            f"{device_type}: {p} fixed at {vals.iloc[0]} in manifest, "
            f"but metadata says {expected}")
