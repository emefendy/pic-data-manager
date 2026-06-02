"""
extract_sim_data.py
-------------------
Converts Tidy3D HDF5 simulation results into design records compatible
with the pic_data.db schema. Supports Y-junction and Euler waveguide bend.

Usage:
    python3 scripts/extract_sim_data.py \
        --sim-type y_junction \
        --hdf5 data/simulation_data.hdf5 \
        --design-id D_YJUNC_001 \
        --params '{"w1": 0.5, "w7": 1.4, "l_junction": 2.0, "t": 0.22}'

    python3 scripts/extract_sim_data.py \
        --sim-type euler_bend \
        --hdf5-circular data/simulation_data_circular.hdf5 \
        --hdf5-euler data/simulation_data_euler.hdf5 \
        --design-id D_EULER_001 \
        --params '{"R_eff": 4.0, "A": 2.4, "t": 0.21, "w": 0.4}'

Output:
    Appends one record to data/designs.json (creates file if missing).
    Also writes the full wavelength-resolved spectrum to data/spectra/<design_id>.json
    so it can be plotted in the dashboard.
"""

import argparse
import json
import numpy as np
import tidy3d as td
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SPECTRA_DIR = DATA_DIR / "spectra"
SPECTRA_DIR.mkdir(parents=True, exist_ok=True)


# ── Extraction functions ──────────────────────────────────────────────────────

def extract_y_junction(hdf5_path: str) -> dict:
    """
    Reads a Y-junction simulation result and returns scalar + spectral metrics.

    Key physics:
      - ModeMonitor "mode" captures amplitude at the top output waveguide.
      - By symmetry, the bottom arm carries identical power → T_total = 2 * T_top.
      - Insertion loss = -10*log10(T_total), reported as a positive dB value.
    """
    sim_data = td.SimulationData.from_file(hdf5_path)
    ldas = td.C_0 / sim_data["mode"].amps.coords["f"].values  # wavelengths in µm

    amp = sim_data["mode"].amps.sel(mode_index=0, direction="+")
    T_top = np.abs(amp.values) ** 2
    T_total = 2 * T_top
    insertion_loss_spectrum = -10 * np.log10(T_total)  # positive dB, lower is better

    return {
        "scalars": {
            # Mean insertion loss across the band — primary KPI
            "simulated_loss_db": float(np.mean(insertion_loss_spectrum)),
            # Bandwidth: wavelength range where loss stays within 0.5 dB of minimum
            "simulated_bandwidth_ghz": _loss_bandwidth_ghz(ldas, insertion_loss_spectrum, margin_db=0.5),
            "min_loss_db": float(np.min(insertion_loss_spectrum)),
            "max_loss_db": float(np.max(insertion_loss_spectrum)),
        },
        "spectrum": {
            "wavelengths_um": ldas.tolist(),
            "T_total": T_total.tolist(),
            "insertion_loss_db": insertion_loss_spectrum.tolist(),
        },
    }


def extract_euler_bend(hdf5_circular: str, hdf5_euler: str) -> dict:
    """
    Reads circular + Euler bend results and returns metrics for both.
    The Euler bend is the primary design; circular is the baseline.

    Key physics:
      - ModeMonitor "mode" at output. Loss = -10*log10(|amp|²).
      - Euler bend has lower loss because curvature transitions smoothly.
    """
    def _extract(path):
        sim_data = td.SimulationData.from_file(path)
        ldas = td.C_0 / sim_data["mode"].amps.coords["f"].values
        amp = sim_data["mode"].amps.sel(mode_index=0, direction="+")
        T = np.abs(amp.values) ** 2
        loss_db = -10 * np.log10(T)
        return ldas, T, loss_db

    ldas, T_euler, loss_euler = _extract(hdf5_euler)
    _, T_circ, loss_circ = _extract(hdf5_circular)

    return {
        "scalars": {
            "simulated_loss_db": float(np.mean(loss_euler)),
            "simulated_bandwidth_ghz": _loss_bandwidth_ghz(ldas, loss_euler, margin_db=0.01),
            "loss_vs_circular_improvement_db": float(np.mean(loss_circ) - np.mean(loss_euler)),
            "circular_loss_db": float(np.mean(loss_circ)),
        },
        "spectrum": {
            "wavelengths_um": ldas.tolist(),
            "T_euler": T_euler.tolist(),
            "T_circular": T_circ.tolist(),
            "loss_euler_db": loss_euler.tolist(),
            "loss_circular_db": loss_circ.tolist(),
        },
    }


def _loss_bandwidth_ghz(ldas_um: np.ndarray, loss_db: np.ndarray, margin_db: float) -> float:
    """
    Returns the wavelength bandwidth (converted to GHz equivalent) over which
    insertion loss stays within `margin_db` of the minimum.
    """
    threshold = np.min(loss_db) + margin_db
    in_band = ldas_um[loss_db <= threshold]
    if len(in_band) < 2:
        return 0.0
    lda_min, lda_max = np.min(in_band), np.max(in_band)
    # Convert wavelength bandwidth to frequency bandwidth in GHz
    f_min = td.C_0 / lda_max  # THz
    f_max = td.C_0 / lda_min
    return float((f_max - f_min) * 1e3)  # THz → GHz


# ── Schema mapping ────────────────────────────────────────────────────────────

SIM_TYPE_CONFIGS = {
    "y_junction": {
        "device_type": "y_junction",
        "required_params": ["w1", "w7", "l_junction", "t"],
        # Maps geometry params to the design record's expected column names.
        # Use these as-is or map to generic names (e.g. waveguide_width_nm).
        "param_map": {
            "w1":          "waveguide_width_nm",   # input/output width in µm → store as-is
            "l_junction":  "ring_radius_um",        # repurposed column; real meaning stored in sim_params
            "t":           "film_thickness_nm",     # silicon layer thickness
        },
        "target_wavelength_nm": 1550,
    },
    "euler_bend": {
        "device_type": "euler_bend",
        "required_params": ["R_eff", "A", "t", "w"],
        "param_map": {
            "w":    "waveguide_width_nm",
            "R_eff": "ring_radius_um",
            "t":    "film_thickness_nm",
        },
        "target_wavelength_nm": 1550,
    },
}


def build_design_record(design_id: str, sim_type: str, params: dict, scalars: dict) -> dict:
    """
    Assembles a record that matches the designs table schema used by ingest_jsons.py.
    Simulation-specific parameters that don't map to generic columns are stored
    in a `sim_params` JSON string so no schema changes are needed.
    """
    config = SIM_TYPE_CONFIGS[sim_type]

    record = {
        "design_id":             design_id,
        "device_type":           config["device_type"],
        "design_version":        "v1",
        "layout_file":           f"gds/{design_id}.gds",
        "simulation_file":       f"sim/{design_id}_simulation.hdf5",
        "target_wavelength_nm":  config["target_wavelength_nm"],
        "designer":              "simulation_import",

        # KPIs — directly replace the seeded random values
        "simulated_loss_db":       scalars["simulated_loss_db"],
        "simulated_bandwidth_ghz": scalars["simulated_bandwidth_ghz"],

        # All geometry/sim params stored as a JSON blob for full fidelity
        "sim_params": json.dumps({**params, **{k: v for k, v in scalars.items()
                                               if k not in ("simulated_loss_db",
                                                            "simulated_bandwidth_ghz")}}),

        # Map common geometry params to the generic schema columns
        "waveguide_width_nm": params.get(
            next((k for k, v in config["param_map"].items() if v == "waveguide_width_nm"), None), None
        ),
        "coupler_gap_nm":    None,
        "ring_radius_um":    params.get(
            next((k for k, v in config["param_map"].items() if v == "ring_radius_um"), None), None
        ),
    }

    return record


# ── I/O helpers ───────────────────────────────────────────────────────────────

def append_to_designs_json(record: dict):
    path = DATA_DIR / "designs.json"
    if path.exists():
        with open(path) as f:
            records = json.load(f)
    else:
        records = []

    # Replace if design_id already exists, otherwise append
    existing_ids = [r["design_id"] for r in records]
    if record["design_id"] in existing_ids:
        idx = existing_ids.index(record["design_id"])
        records[idx] = record
        print(f"Updated existing record for {record['design_id']}")
    else:
        records.append(record)
        print(f"Appended new record for {record['design_id']}")

    with open(path, "w") as f:
        json.dump(records, f, indent=2)


def save_spectrum(design_id: str, spectrum: dict):
    path = SPECTRA_DIR / f"{design_id}.json"
    with open(path, "w") as f:
        json.dump(spectrum, f, indent=2)
    print(f"Spectrum saved to {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract Tidy3D sim results into pic_data schema.")
    parser.add_argument("--sim-type",  required=True, choices=list(SIM_TYPE_CONFIGS.keys()))
    parser.add_argument("--design-id", required=True)
    parser.add_argument("--params",    required=True, help="JSON string of geometry params")

    # Y-junction
    parser.add_argument("--hdf5", help="Path to HDF5 for single-sim types (y_junction)")

    # Euler bend (two sims)
    parser.add_argument("--hdf5-circular", help="Circular bend HDF5 path")
    parser.add_argument("--hdf5-euler",    help="Euler bend HDF5 path")

    args = parser.parse_args()
    params = json.loads(args.params)

    print(f"\nExtracting {args.sim_type} results for {args.design_id}...")

    if args.sim_type == "y_junction":
        assert args.hdf5, "--hdf5 required for y_junction"
        result = extract_y_junction(args.hdf5)

    elif args.sim_type == "euler_bend":
        assert args.hdf5_circular and args.hdf5_euler, \
            "--hdf5-circular and --hdf5-euler required for euler_bend"
        result = extract_euler_bend(args.hdf5_circular, args.hdf5_euler)

    print(f"  Loss:      {result['scalars']['simulated_loss_db']:.4f} dB")
    print(f"  Bandwidth: {result['scalars']['simulated_bandwidth_ghz']:.1f} GHz")

    record = build_design_record(args.design_id, args.sim_type, params, result["scalars"])
    append_to_designs_json(record)
    save_spectrum(args.design_id, result["spectrum"])

    print(f"\nDone. Re-run ingest_jsons.py → run_ai.py → train_model.py to refresh the dashboard.")


if __name__ == "__main__":
    main()
