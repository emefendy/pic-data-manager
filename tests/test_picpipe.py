"""Validation tests for the picpipe real-data pipeline (pytest -q)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from picpipe import analysis
from picpipe.components import EULER_BEND, Y_JUNCTION
from picpipe.ingest import ingest_real_dataset, ingest_spectrum
from picpipe.models import AnalysisResult, Component, SimulationRun
from picpipe.optimize import optimize_over_real_sweep
from picpipe.store import PicStore

PROJECT = Path(__file__).resolve().parents[1]


def _spectrum(loss_db=0.2, n=101):
    wl = np.linspace(1.50, 1.60, n)
    t = np.full(n, 10 ** (-loss_db / 10.0))
    return pd.DataFrame({"wavelength_um": wl, "transmission": t})


# ----------------------------- analysis ---------------------------------- #
def test_insertion_loss_zero_for_unity():
    assert analysis.mean_insertion_loss_db(np.ones(10)) == pytest.approx(0.0, abs=1e-9)


def test_insertion_loss_3db_for_half():
    assert analysis.mean_insertion_loss_db(np.full(5, 0.5)) == pytest.approx(3.0103, abs=1e-3)


def test_bandwidth_units_are_nm_and_bounded():
    df = _spectrum()
    bw = analysis.bandwidth_3db_nm(df["wavelength_um"].to_numpy(), df["transmission"].to_numpy())
    assert 0 < bw <= 100 + 1e-6  # 1.50–1.60 µm == 100 nm band


def test_sanity_check_flags_nonphysical():
    df = pd.DataFrame({"wavelength_um": [1.5, 1.55, 1.6], "transmission": [0.9, 2.0, 0.9]})
    assert any("exceeds" in w for w in analysis.check_spectrum(df))


def test_analyze_rejects_bad_spectrum():
    df = pd.DataFrame({"wavelength_um": [1.5, 1.55, 1.6], "transmission": [0.9, np.nan, 0.9]})
    with pytest.raises(ValueError):
        analysis.analyze_spectrum("c", "r", df)


# ----------------------------- models ------------------------------------ #
def test_component_rejects_empty_id():
    with pytest.raises(Exception):
        Component(component_id="", type="x", name="n", library="l")


def test_run_requires_increasing_wavelength():
    with pytest.raises(Exception):
        SimulationRun(run_id="r", component_id="c", tool="tidy3d",
                      wl_min_um=1.6, wl_max_um=1.5, n_wavelengths=10)


def test_metric_must_be_finite():
    with pytest.raises(Exception):
        AnalysisResult(result_id="x", run_id="r", component_id="c",
                       metric_name="m", metric_value=float("inf"), units="dB")


# ----------------------------- store + ingest ---------------------------- #
def test_ingest_roundtrip_and_queries(tmp_path):
    s = PicStore(tmp_path / "p")
    rid = ingest_spectrum(s, Y_JUNCTION, _spectrum(0.2), geometry_snapshot={"l_junction": 1.5})
    assert rid in set(s.runs_for_component("y_junction")["run_id"])
    metrics = s.metrics(component_id="y_junction")
    assert {"insertion_loss_mean", "bandwidth_3db"} <= set(metrics["metric_name"])
    back = s.read_spectrum(rid)
    assert np.allclose(back["transmission"], _spectrum(0.2)["transmission"])
    arts = s._df("SELECT * FROM raw_artifacts")
    assert arts.iloc[0]["checksum_sha256"]
    s.close()


def test_ingest_idempotent(tmp_path):
    s = PicStore(tmp_path / "p")
    a = ingest_spectrum(s, Y_JUNCTION, _spectrum(0.2), geometry_snapshot={"l_junction": 1.5})
    b = ingest_spectrum(s, Y_JUNCTION, _spectrum(0.2), geometry_snapshot={"l_junction": 1.5})
    assert a == b
    assert len(s.runs_for_component("y_junction")) == 1
    s.close()


def test_filter_by_geometry(tmp_path):
    s = PicStore(tmp_path / "p")
    for L in (1.0, 1.5, 2.0):
        ingest_spectrum(s, Y_JUNCTION, _spectrum(0.2 + L * 0.01),
                        geometry_snapshot={"l_junction": L}, run_suffix=f"L{L}")
    sel = s.filter_runs_by_geometry("l_junction", min_val=1.2, max_val=1.8)
    assert len(sel) == 1 and float(sel.iloc[0]["geom_value"]) == pytest.approx(1.5)
    s.close()


def test_optimization_records_and_improvement(tmp_path):
    s = PicStore(tmp_path / "p")
    # longer junction -> lower loss (mirrors the real trend)
    for L, loss in [(1.0, 0.32), (1.5, 0.23), (2.0, 0.16)]:
        ingest_spectrum(s, Y_JUNCTION, _spectrum(loss),
                        geometry_snapshot={"l_junction": L}, run_suffix=f"L{L}")
    opt = optimize_over_real_sweep(s, component_id="y_junction", param="l_junction")
    assert opt["n_candidates"] == 3
    assert opt["best_param"] == 2.0
    assert opt["improvement_db"] > 0
    hist = s.optimization_history(opt["study_id"])
    assert len(hist) == 3 and bool(hist["is_best"].iloc[-1])
    s.close()


# ----------------------------- real-data integration --------------------- #
@pytest.mark.skipif(not (PROJECT / "pic_data.db").exists(),
                    reason="pic_data.db not present")
def test_real_dataset_ingests(tmp_path):
    s = PicStore(tmp_path / "p")
    ids = ingest_real_dataset(s, str(PROJECT / "pic_data.db"), str(PROJECT / "data" / "spectra"))
    assert len(ids) > 0
    # both reference components present, with real (non-synthetic) runs
    types = set(s.components()["type"])
    assert {"y_junction", "euler_bend"} & types
    assert not s.runs_for_component("y_junction")["is_synthetic"].any()
    s.close()
