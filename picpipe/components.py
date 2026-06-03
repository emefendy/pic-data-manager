"""Reference component metadata records (Task 3).

Two components, matching the two device families this project actually simulated
with Tidy3D FDTD (see scripts/sweep_yjunction.py and scripts/sweep_eulerbend.py):

  * Y-junction 1x2 power splitter (adapted from the Tidy3D example library).
  * Euler (clothoid) waveguide bend, benchmarked against a circular bend.

Each record carries the minimal metadata the challenge asks for: name, function,
geometry parameters, material/platform assumptions, and expected performance.
Every real swept design becomes a SimulationRun (variant) of one of these
components, so "all simulations for a component" and "compare variants" work.
"""

from __future__ import annotations

from picpipe.models import Component

Y_JUNCTION = Component(
    component_id="y_junction",
    type="y_junction",
    name="Compact SOI Y-junction (1x2 splitter)",
    library="Tidy3D example library",
    source_url="https://www.flexcompute.com/tidy3d/examples/notebooks/YJunction/",
    function="1x2 equal power splitter",
    platform="SOI-220nm",
    materials={"core": "cSi (n≈3.476)", "cladding": "SiO2 (n≈1.444)"},
    geometry={
        "thickness_um": 0.22,
        "input_width_um": 0.5,
        "swept_params": ["w1", "w7", "l_junction"],
        "junction_length_um": "swept (≈1.0–2.0)",
    },
    expected_metrics={"insertion_loss_mean_db": 0.2, "band_nm": [1500, 1600]},
    notes="Adapted from Zhang et al., 'A compact and low loss Y-junction for "
          "submicron silicon waveguide,' Opt. Express 21, 1310-1316 (2013). "
          "Real FDTD sweep: scripts/sweep_yjunction.py.",
)

EULER_BEND = Component(
    component_id="euler_bend",
    type="euler_bend",
    name="Euler (clothoid) waveguide bend",
    library="Public SOI reference design",
    source_url="https://gdsfactory.github.io/gdsfactory/",
    function="Low-loss 90° waveguide bend (vs circular baseline)",
    platform="SOI-220nm",
    materials={"core": "cSi (n≈3.476)", "cladding": "SiO2 (n≈1.444)"},
    geometry={
        "thickness_um": 0.22,
        "waveguide_width_um": 0.5,
        "swept_params": ["R_eff", "A"],
        "effective_radius_um": "swept",
    },
    expected_metrics={"insertion_loss_db": 0.05, "improves_on": "circular bend"},
    notes="Euler/clothoid bend benchmarked against a circular bend of equal "
          "effective radius. Real FDTD sweep: scripts/sweep_eulerbend.py.",
)

# device_type (from pic_data.db) -> Component
COMPONENTS = {c.type: c for c in (Y_JUNCTION, EULER_BEND)}
ALL_COMPONENTS = list(COMPONENTS.values())

# Which spectrum key holds the transmission curve to analyze, per device type.
TRANSMISSION_KEY = {"y_junction": "T_total", "euler_bend": "T_euler"}
