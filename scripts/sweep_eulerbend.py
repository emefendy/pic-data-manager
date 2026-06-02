"""
sweep_eulerbend.py
------------------
Runs 30 Euler waveguide bend FDTD simulations with varied geometry parameters,
downloads each result, and ingests them into data/designs.json.

Varies: R_eff (effective radius), A (clothoid parameter), w (waveguide width)
5 x 3 x 2 = 30 combinations.

Usage (from project root):
    python3 scripts/sweep_eulerbend.py

Each job runs both a circular bend baseline AND the Euler bend (2 jobs per combo).
Estimated cost: ~60 x 0.042 = 2.5 FlexCredits total.
"""

import json
import itertools
import numpy as np
import scipy.integrate as integrate
from pathlib import Path
from scipy.optimize import fsolve
import tidy3d as td
import tidy3d.web as web

ROOT       = Path(__file__).resolve().parents[1]
DATA_DIR   = ROOT / "data"
SPECTRA_DIR = DATA_DIR / "spectra"
HDF5_DIR   = ROOT / "notebooks" / "data" / "sweep_euler"
SPECTRA_DIR.mkdir(parents=True, exist_ok=True)
HDF5_DIR.mkdir(parents=True, exist_ok=True)

# ── Parameter sweep ───────────────────────────────────────────────────────────
# 5 x 3 x 2 = 30 combinations

R_EFF_VALUES = [3.0, 3.5, 4.0, 4.5, 5.0]   # effective bend radius (µm)
A_VALUES     = [2.0, 2.2, 2.4]              # clothoid parameter
W_VALUES     = [0.38, 0.42]                 # waveguide width (µm)

COMBOS = list(itertools.product(R_EFF_VALUES, A_VALUES, W_VALUES))
assert len(COMBOS) == 30

# Fixed
T      = 0.21
N_SI   = 3.476
N_SIO2 = 1.444
INF_EFF = 100


# ── Euler curve geometry ──────────────────────────────────────────────────────

def euler_curve(L_max, A, num_points=50):
    Ls = np.linspace(0, L_max, num_points)
    x1 = np.array([integrate.quad(lambda th: A * np.cos(th**2 / 2), 0, L / A)[0] for L in Ls])
    y1 = np.array([integrate.quad(lambda th: A * np.sin(th**2 / 2), 0, L / A)[0] for L in Ls])
    return x1, y1


def calculate_L_max(R_eff, A, num_points=50, tolerance=1e-8):
    def func(L_max):
        L_max = float(np.atleast_1d(L_max)[0])
        x1, y1 = euler_curve(L_max, A, num_points)
        k  = -(x1[-1] - x1[-2]) / (y1[-1] - y1[-2])
        xp, yp = x1[-1], y1[-1]
        R  = np.sqrt(
            ((R_eff + k * xp - yp) / (k + 1) - xp) ** 2
            + (-(R_eff + k * xp - yp) / (k + 1) + R_eff - yp) ** 2
        )
        return np.abs(R - A**2 / L_max)
    res = fsolve(func, x0=(1,), full_output=True)
    error = float(res[1]["fvec"])
    if error > tolerance:
        raise ValueError(f"L_max solver did not converge for R_eff={R_eff}, A={A} (residual={error:.2e})")
    return float(res[0][0])


def build_euler_coords(R_eff, A):
    L_max = calculate_L_max(R_eff, A)
    x1, y1 = euler_curve(L_max, A)
    xp, yp = x1[-1], y1[-1]
    R_min = A**2 / L_max

    x3 = np.flipud(R_eff - y1)
    y3 = np.flipud(R_eff - x1)

    def circle_eq(var):
        a, b = var
        return [
            (xp - a)**2 + (yp - b)**2 - R_min**2,
            (R_eff - yp - a)**2 + (R_eff - xp - b)**2 - R_min**2,
        ]
    sol = fsolve(circle_eq, (0, R_eff), full_output=True)
    a, b = sol[0]
    residual = np.max(np.abs(sol[1]["fvec"]))
    if residual > 1e-6:
        raise ValueError(f"Circle solver did not converge for R_eff={R_eff}, A={A} (residual={residual:.2e})")
    x2 = np.linspace(xp + 0.01, R_eff - yp - 0.01, 50)
    y2 = -np.sqrt(np.clip(R_min**2 - (x2 - a)**2, 0, None)) + b

    x_euler = np.concatenate((x1, x2, x3))
    y_euler = np.concatenate((y1, y2, y3))

    x_circle = np.linspace(0, R_eff, 100)
    y_circle = -np.sqrt(np.clip(R_eff**2 - x_circle**2, 0, None)) + R_eff

    return x_euler, y_euler, x_circle, y_circle


def bend_to_structure(x, y, w, t, R_eff):
    import gdstk
    cell = gdstk.Cell("bend")
    x_ext = np.concatenate([[-INF_EFF], x, [R_eff]])
    y_ext = np.concatenate([[0],        y, [INF_EFF]])
    cell.add(gdstk.FlexPath(x_ext + 1j * y_ext, w, layer=1, datatype=0))
    return td.Structure(
        geometry=td.PolySlab.from_gds(cell, gds_layer=1, axis=2, slab_bounds=(0, t))[0],
        medium=td.Medium(permittivity=N_SI**2),
    )


# ── Simulation builder ────────────────────────────────────────────────────────

def build_sim(x_bend, y_bend, w, R_eff):
    si   = td.Medium(permittivity=N_SI**2)
    sio2 = td.Medium(permittivity=N_SIO2**2)

    lda0   = 1.55
    freq0  = td.C_0 / lda0
    ldas   = np.linspace(1.5, 1.6, 100)
    freqs  = td.C_0 / ldas
    fwidth = 0.5 * (np.max(freqs) - np.min(freqs))

    buffer = 1.0
    bend   = bend_to_structure(x_bend, y_bend, w, T, R_eff)

    mode_spec = td.ModeSpec(num_modes=1, target_neff=N_SI)

    mode_source = td.ModeSource(
        center=(-buffer, 0, T / 2),
        size=(0, 4 * w, 6 * T),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
        direction="+",
        mode_spec=mode_spec,
        mode_index=0,
    )
    mode_monitor = td.ModeMonitor(
        center=(R_eff, R_eff + buffer, T / 2),
        size=(4 * w, 0, 6 * T),
        freqs=freqs,
        mode_spec=mode_spec,
        name="mode",
    )

    sim = td.Simulation(
        center=(R_eff / 2, R_eff / 2, T / 2),
        size=(R_eff + 2 * w + lda0, R_eff + 2 * w + lda0, 10 * T),
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=20, wavelength=lda0),
        structures=[bend],
        sources=[mode_source],
        monitors=[mode_monitor],
        run_time=5e-13,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
        medium=sio2,
    )
    return sim, freqs


# ── Extraction + ingestion ────────────────────────────────────────────────────

def extract_and_ingest(design_id, sim_data_euler, sim_data_circular, freqs, params):
    ldas = td.C_0 / freqs

    def get_loss(sim_data):
        amp = sim_data["mode"].amps.sel(mode_index=0, direction="+")
        T   = np.abs(amp.values) ** 2
        return T, -10 * np.log10(T)

    T_euler,  loss_euler  = get_loss(sim_data_euler)
    T_circular, loss_circ = get_loss(sim_data_circular)

    def bw_ghz(loss_db):
        threshold = np.min(loss_db) + 0.01
        in_band   = ldas[loss_db <= threshold]
        if len(in_band) < 2:
            return 0.0
        f_min = td.C_0 / np.max(in_band)
        f_max = td.C_0 / np.min(in_band)
        return float((f_max - f_min) * 1e3)

    scalars = {
        "simulated_loss_db":              float(np.mean(loss_euler)),
        "simulated_bandwidth_ghz":        bw_ghz(loss_euler),
        "circular_loss_db":               float(np.mean(loss_circ)),
        "loss_improvement_vs_circular_db": float(np.mean(loss_circ) - np.mean(loss_euler)),
        "min_loss_db":                    float(np.min(loss_euler)),
    }

    spectrum = {
        "wavelengths_um":    ldas.tolist(),
        "T_euler":           T_euler.tolist(),
        "T_circular":        T_circular.tolist(),
        "loss_euler_db":     loss_euler.tolist(),
        "loss_circular_db":  loss_circ.tolist(),
    }
    (SPECTRA_DIR / f"{design_id}.json").write_text(json.dumps(spectrum, indent=2))

    record = {
        "design_id":               design_id,
        "device_type":             "euler_bend",
        "design_version":          "v1",
        "layout_file":             f"gds/{design_id}.gds",
        "simulation_file":         f"notebooks/data/sweep_euler/{design_id}.hdf5",
        "target_wavelength_nm":    1550,
        "designer":                "sweep",
        "simulated_loss_db":       scalars["simulated_loss_db"],
        "simulated_bandwidth_ghz": scalars["simulated_bandwidth_ghz"],
        "waveguide_width_nm":      params["w"] * 1000,
        "coupler_gap_nm":          None,
        "ring_radius_um":          params["R_eff"],
        "sim_params":              json.dumps({**params, **scalars}),
    }

    path    = DATA_DIR / "designs.json"
    records = json.loads(path.read_text()) if path.exists() else []
    ids     = [r["design_id"] for r in records]
    if design_id in ids:
        records[ids.index(design_id)] = record
    else:
        records.append(record)
    path.write_text(json.dumps(records, indent=2))

    print(
        f"  euler={scalars['simulated_loss_db']:.4f} dB  "
        f"circ={scalars['circular_loss_db']:.4f} dB  "
        f"improvement={scalars['loss_improvement_vs_circular_db']:.4f} dB  → ingested"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load already-ingested design IDs so we can skip re-running them
    designs_path = DATA_DIR / "designs.json"
    existing_ids = set()
    if designs_path.exists():
        existing_ids = {r["design_id"] for r in json.loads(designs_path.read_text())}

    print(f"Submitting up to {len(COMBOS) * 2} jobs (euler + circular per combo)...\n")

    jobs = []

    for i, (R_eff, A, w) in enumerate(COMBOS):
        design_id = f"D_EB_{i+1:03d}"

        if design_id in existing_ids:
            print(f"  {design_id}  SKIPPED (already ingested)")
            continue

        print(f"  Building {design_id}  R_eff={R_eff}, A={A}, w={w}...", end=" ")

        try:
            x_euler, y_euler, x_circle, y_circle = build_euler_coords(R_eff, A)
        except Exception as e:
            print(f"SKIPPED (geometry failed: {e})")
            continue

        try:
            sim_euler,    freqs = build_sim(x_euler,  y_euler,  w, R_eff)
            sim_circular, _     = build_sim(x_circle, y_circle, w, R_eff)
        except Exception as e:
            print(f"SKIPPED (sim build failed: {e})")
            continue

        job_euler    = web.Job(simulation=sim_euler,    task_name=f"{design_id}_euler",    verbose=False)
        job_circular = web.Job(simulation=sim_circular, task_name=f"{design_id}_circular", verbose=False)

        jobs.append((design_id, job_euler, job_circular, freqs,
                     {"R_eff": R_eff, "A": A, "w": w, "t": T}))
        print("submitted")

    print(f"\nAll jobs submitted. Downloading results...\n")

    for design_id, job_euler, job_circular, freqs, params in jobs:
        print(f"Downloading {design_id}...", end=" ", flush=True)
        try:
            sd_euler    = job_euler.run(path=str(HDF5_DIR / f"{design_id}_euler.hdf5"))
            sd_circular = job_circular.run(path=str(HDF5_DIR / f"{design_id}_circular.hdf5"))
            extract_and_ingest(design_id, sd_euler, sd_circular, freqs, params)
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\nDone. Refresh the dashboard:")
    print("  python3 scripts/ingest_jsons.py")
    print("  python3 scripts/run_ai.py")
    print("  python3 scripts/analyze_performance.py")
    print("  python3 scripts/train_model.py")


if __name__ == "__main__":
    main()
