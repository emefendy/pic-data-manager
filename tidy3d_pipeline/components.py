from __future__ import annotations

from tidy3d_pipeline.models import Component

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
        "swept_params": ["w7", "l_junction", "h_bend"],
        "junction_output_width_w7_um": "swept (1.2–1.6)",
        "junction_length_um": "swept (1.5 or 2.0)",
        "s_bend_offset_h_bend_um": "swept (1.5–2.5)",
        "fixed_params": {"input_width_w1_um": 0.5, "thickness_t_um": 0.22},
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
    platform="SOI (0.21 µm core)",
    materials={"core": "cSi (n≈3.476)", "cladding": "SiO2 (n≈1.444)"},
    geometry={
        "swept_params": ["R_eff", "A", "w"],
        "effective_radius_um": "swept (3.0–5.0)",
        "clothoid_A_um": "swept (2.0–2.8)",
        "waveguide_width_um": "swept (0.38 or 0.42)",
        "fixed_params": {"thickness_t_um": 0.21},
    },
    expected_metrics={"insertion_loss_db": 0.05, "improves_on": "circular bend"},
    notes="Euler/clothoid bend benchmarked against a circular bend of equal "
          "effective radius. Real FDTD sweep: scripts/sweep_eulerbend.py.",
)

COMPONENTS = {c.type: c for c in (Y_JUNCTION, EULER_BEND)}
ALL_COMPONENTS = list(COMPONENTS.values())

TRANSMISSION_KEY = {"y_junction": "T_total", "euler_bend": "T_euler"}
