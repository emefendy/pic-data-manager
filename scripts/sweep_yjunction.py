"""
sweep_yjunction.py
------------------
Runs 30 Y-junction FDTD simulations with varied geometry parameters,
downloads each result, and ingests them all into data/designs.json.

Usage (from project root):
    python3 scripts/sweep_yjunction.py

Requires a configured Tidy3D API key (~/.tidy3d/config).
Estimated cost: ~30 x 0.116 = 3.5 FlexCredits.
Each job takes ~1-2 minutes. Total wall time: ~30-60 min running in parallel.
"""

import json
import time
import itertools
import numpy as np
from pathlib import Path
import tidy3d as td
import tidy3d.web as web

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SPECTRA_DIR = DATA_DIR / "spectra"
HDF5_DIR = ROOT / "notebooks" / "data" / "sweep"
SPECTRA_DIR.mkdir(parents=True, exist_ok=True)
HDF5_DIR.mkdir(parents=True, exist_ok=True)

# ── Parameter sweep ───────────────────────────────────────────────────────────
# Vary three key parameters around the optimized baseline.
# 5 x 3 x 2 = 30 combinations exactly.

W7_VALUES   = [1.2, 1.3, 1.4, 1.5, 1.6]   # center junction width (µm)
H_BEND_VALUES = [1.5, 2.0, 2.5]            # vertical bend offset (µm)
L_JUNCTION_VALUES = [1.5, 2.0]             # junction length (µm)

COMBOS = list(itertools.product(W7_VALUES, H_BEND_VALUES, L_JUNCTION_VALUES))
assert len(COMBOS) == 30, f"Expected 30 combos, got {len(COMBOS)}"

# Fixed parameters (baseline optimized design)
T       = 0.22
W1      = 0.5
W2      = 0.5
W3      = 0.6
W4      = 0.7
W5      = 0.9
W6      = 1.26
W8      = 1.4
W9      = 1.4
W10     = 1.4
W11     = 1.31
W12     = 1.2
W13     = 1.2
L_IN    = 1.0
L_BEND  = 6.0
L_OUT   = 1.0
INF_EFF = 100


# ── Simulation builder ────────────────────────────────────────────────────────

def build_simulation(w7: float, h_bend: float, l_junction: float) -> td.Simulation:
    si   = td.material_library["cSi"]["Palik_Lossless"]
    sio2 = td.material_library["SiO2"]["Palik_Lossless"]

    lda0  = 1.55
    freq0 = td.C_0 / lda0
    ldas  = np.linspace(1.5, 1.6, 101)
    freqs = td.C_0 / ldas
    fwidth = 0.5 * (np.max(freqs) - np.min(freqs))

    # Junction polygon — use w7 as the swept center width
    widths = np.array([W1, W2, W3, W4, W5, W6, w7, W8, W9, W10, W11, W12, W13])
    x = np.linspace(L_IN, L_IN + l_junction, 13)
    x = np.concatenate((x, np.flipud(x)))
    y = np.concatenate((widths / 2, -np.flipud(widths / 2)))
    vertices = np.transpose(np.vstack((x, y)))

    junction = td.Structure(
        geometry=td.PolySlab(vertices=vertices, axis=2, slab_bounds=(0, T)),
        medium=si,
    )

    # S-bend waveguides (sine curve, no photonforge needed)
    x_start = L_IN + l_junction
    n_pts   = 200

    def sbend_vertices(sign):
        """Generate polygon vertices for one S-bend arm + trailing straight."""
        xs = np.linspace(0, L_BEND, n_pts)
        ys = (xs * h_bend / L_BEND
              - h_bend / (2 * np.pi) * np.sin(2 * np.pi * xs / L_BEND))
        ys *= sign

        # centerline offset: arm starts at ±(W13/2 - W1/2)
        y_offset = sign * (W13 / 2 - W1 / 2)
        ys += y_offset

        # extend to effective infinity
        xs_full = np.append(xs + x_start, x_start + L_BEND + INF_EFF)
        ys_full = np.append(ys, ys[-1])

        # build closed polygon with waveguide width W1
        top_x = xs_full
        top_y = ys_full + W1 / 2
        bot_x = np.flipud(xs_full)
        bot_y = np.flipud(ys_full) - W1 / 2
        verts = np.column_stack([
            np.concatenate([top_x, bot_x]),
            np.concatenate([top_y, bot_y]),
        ])
        return verts

    wg_bend_1 = td.Structure(
        geometry=td.PolySlab(vertices=sbend_vertices(+1), axis=2, slab_bounds=(0, T)),
        medium=si,
    )
    wg_bend_2 = td.Structure(
        geometry=td.PolySlab(vertices=sbend_vertices(-1), axis=2, slab_bounds=(0, T)),
        medium=si,
    )

    wg_in = td.Structure(
        geometry=td.Box.from_bounds(
            rmin=(-INF_EFF, -W1 / 2, 0), rmax=(L_IN, W1 / 2, T)
        ),
        medium=si,
    )

    Lx = L_IN + l_junction + L_OUT + L_BEND
    Ly = W13 + 2 * h_bend + 1.5 * lda0
    Lz = 10 * T

    mode_spec = td.ModeSpec(num_modes=1, target_neff=3.5)

    mode_source = td.ModeSource(
        center=(L_IN / 2, 0, T / 2),
        size=(0, 4 * W1, 6 * T),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
        direction="+",
        mode_spec=mode_spec,
        mode_index=0,
    )

    mode_monitor = td.ModeMonitor(
        center=(L_IN + l_junction + L_BEND + L_OUT / 2,
                W13 / 2 - W1 / 2 + h_bend,
                T / 2),
        size=(0, 4 * W1, 6 * T),
        freqs=freqs,
        mode_spec=mode_spec,
        name="mode",
    )

    sim = td.Simulation(
        center=(Lx / 2, 0, 0),
        size=(Lx, Ly, Lz),
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=20, wavelength=lda0),
        structures=[wg_in, junction, wg_bend_1, wg_bend_2],
        sources=[mode_source],
        monitors=[mode_monitor],
        run_time=5e-13,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
        medium=sio2,
    )

    return sim, freqs


# ── Result extraction ─────────────────────────────────────────────────────────

def extract_and_ingest(design_id, sim_data, freqs, params):
    ldas = td.C_0 / freqs

    amp  = sim_data["mode"].amps.sel(mode_index=0, direction="+")
    T_top   = np.abs(amp.values) ** 2
    T_total = 2 * T_top
    loss_db = -10 * np.log10(T_total)

    # bandwidth: range where loss within 0.5 dB of minimum
    threshold = np.min(loss_db) + 0.5
    in_band   = ldas[loss_db <= threshold]
    if len(in_band) >= 2:
        f_min = td.C_0 / np.max(in_band)  # Hz (Tidy3D units: length in µm)
        f_max = td.C_0 / np.min(in_band)  # Hz
        bw_ghz = float((f_max - f_min) / 1e9)  # Hz → GHz
    else:
        bw_ghz = 0.0

    scalars = {
        "simulated_loss_db":       float(np.mean(loss_db)),
        "simulated_bandwidth_ghz": bw_ghz,
        "min_loss_db":             float(np.min(loss_db)),
        "max_loss_db":             float(np.max(loss_db)),
    }

    # Save spectrum
    spectrum = {
        "wavelengths_um":   ldas.tolist(),
        "T_total":          T_total.tolist(),
        "insertion_loss_db": loss_db.tolist(),
    }
    (SPECTRA_DIR / f"{design_id}.json").write_text(json.dumps(spectrum, indent=2))

    # Build design record
    record = {
        "design_id":               design_id,
        "device_type":             "y_junction",
        "design_version":          "v1",
        "layout_file":             f"gds/{design_id}.gds",
        "simulation_file":         f"notebooks/data/sweep/{design_id}.hdf5",
        "target_wavelength_nm":    1550,
        "designer":                "sweep",
        "simulated_loss_db":       scalars["simulated_loss_db"],
        "simulated_bandwidth_ghz": scalars["simulated_bandwidth_ghz"],
        "waveguide_width_nm":      params["w1"] * 1000,
        "coupler_gap_nm":          None,
        "ring_radius_um":          params["l_junction"],
        "sim_params":              json.dumps({**params, **scalars}),
    }

    # Append/update designs.json
    path = DATA_DIR / "designs.json"
    records = json.loads(path.read_text()) if path.exists() else []
    ids = [r["design_id"] for r in records]
    if design_id in ids:
        records[ids.index(design_id)] = record
    else:
        records.append(record)
    path.write_text(json.dumps(records, indent=2))

    print(f"  loss={scalars['simulated_loss_db']:.4f} dB  bw={bw_ghz:.1f} GHz  → ingested")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Submitting {len(COMBOS)} jobs to Tidy3D...\n")

    jobs = []
    freqs_list = []

    for i, (w7, h_bend, l_junction) in enumerate(COMBOS):
        design_id = f"D_YJ_{i+1:03d}"
        sim, freqs = build_simulation(w7, h_bend, l_junction)
        job = web.Job(simulation=sim, task_name=design_id, verbose=False)
        jobs.append((design_id, job, freqs, {"w1": W1, "w7": w7, "h_bend": h_bend,
                                              "l_junction": l_junction, "t": T}))
        freqs_list.append(freqs)
        print(f"  Submitted {design_id}  (w7={w7}, h_bend={h_bend}, l_junction={l_junction})")

    print(f"\nAll jobs submitted. Waiting for results...\n")

    for design_id, job, freqs, params in jobs:
        hdf5_path = str(HDF5_DIR / f"{design_id}.hdf5")
        print(f"Downloading {design_id}...", end=" ", flush=True)
        try:
            sim_data = job.run(path=hdf5_path)
            extract_and_ingest(design_id, sim_data, freqs, params)
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\nDone. Run the pipeline to refresh the dashboard:")
    print("  python3 scripts/ingest_jsons.py")
    print("  python3 scripts/run_ai.py")
    print("  python3 scripts/analyze_performance.py")
    print("  python3 scripts/train_model.py")


if __name__ == "__main__":
    main()
