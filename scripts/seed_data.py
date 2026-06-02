import json
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

np.random.seed(42)


def to_json_records(df: pd.DataFrame, filename: str):
    path = DATA_DIR / filename
    records = df.to_dict(orient="records")

    with open(path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Wrote {len(records)} records to {path}")


def main():
    num_designs = 30
    wafers_per_design = 3
    devices_per_wafer = 30

    device_types = [
        "ring_resonator",
        "mach_zehnder",
        "grating_coupler",
        "directional_coupler",
    ]

    foundries = ["Foundry_X", "Foundry_Y", "Foundry_Z"]
    test_stations = ["optical_probe_station_1", "optical_probe_station_2"]
    operators = ["operator_a", "operator_b", "operator_c"]

    designs = pd.DataFrame({
        "design_id": [f"D{i:03d}" for i in range(num_designs)],
        "device_type": np.random.choice(device_types, size=num_designs),
        "design_version": np.random.choice(["v1", "v2", "v3"], size=num_designs),
        "layout_file": [f"gds/design_D{i:03d}.gds" for i in range(num_designs)],
        "simulation_file": [f"sim/design_D{i:03d}_simulation.json" for i in range(num_designs)],
        "waveguide_width_nm": np.random.normal(480, 35, size=num_designs).round(1),
        "coupler_gap_nm": np.random.normal(230, 25, size=num_designs).round(1),
        "ring_radius_um": np.random.normal(12, 3, size=num_designs).round(1),
        "target_wavelength_nm": np.random.choice([1310, 1550], size=num_designs),
        "simulated_loss_db": np.random.normal(2.0, 0.4, size=num_designs).round(2),
        "simulated_bandwidth_ghz": np.random.normal(48, 7, size=num_designs).round(2),
        "designer": np.random.choice(["designer_a", "designer_b"], size=num_designs),
    })

    fabrication_rows = []

    for _, design in designs.iterrows():
        for wafer_num in range(wafers_per_design):
            foundry = np.random.choice(foundries)

            foundry_width_bias = {
                "Foundry_X": 2,
                "Foundry_Y": 12,
                "Foundry_Z": -8,
            }[foundry]

            foundry_defect_lambda = {
                "Foundry_X": 4,
                "Foundry_Y": 7,
                "Foundry_Z": 5,
            }[foundry]

            measured_width = (
                design["waveguide_width_nm"]
                + foundry_width_bias
                + np.random.normal(0, 15)
            )

            wafer_id = f"W_{design['design_id']}_{wafer_num}"

            fabrication_rows.append({
                "wafer_id": wafer_id,
                "fab_run": np.random.choice(["RUN_A", "RUN_B", "RUN_C"]),
                "foundry": foundry,
                "design_id": design["design_id"],
                "process_recipe": np.random.choice(["recipe_alpha", "recipe_beta"]),
                "film_thickness_nm": round(np.random.normal(220, 5), 2),
                "etch_depth_nm": round(np.random.normal(70, 4), 2),
                "measured_width_nm": round(measured_width, 2),
                "width_error_nm": round(measured_width - design["waveguide_width_nm"], 2),
                "sidewall_roughness_nm": round(np.random.normal(3.5, 0.8), 2),
                "defect_count": int(np.random.poisson(foundry_defect_lambda)),
                "metrology_file": f"metrology/{wafer_id}_metrology.csv",
                "fabrication_date": (
                    datetime(2026, 5, 1) + timedelta(days=int(np.random.randint(0, 20)))
                ).date().isoformat(),
            })

    fabrication = pd.DataFrame(fabrication_rows)

    test_rows = []

    for _, fab in fabrication.iterrows():
        design = designs[designs["design_id"] == fab["design_id"]].iloc[0]

        for device_num in range(devices_per_wafer):
            device_id = f"DEV_{fab['wafer_id']}_{device_num:03d}"

            width_error = abs(fab["measured_width_nm"] - design["waveguide_width_nm"])
            defect_penalty = fab["defect_count"] * 0.06
            roughness_penalty = fab["sidewall_roughness_nm"] * 0.18

            insertion_loss = (
                design["simulated_loss_db"]
                + 0.012 * width_error
                + defect_penalty
                + roughness_penalty
                + np.random.normal(0, 0.35)
            )

            extinction_ratio = (
                23
                - 0.035 * width_error
                - fab["defect_count"] * 0.12
                - fab["sidewall_roughness_nm"] * 0.3
                + np.random.normal(0, 2.0)
            )

            bandwidth = (
                design["simulated_bandwidth_ghz"]
                - 0.10 * width_error
                - fab["defect_count"] * 0.25
                + np.random.normal(0, 5.0)
            )

            passed = (
                insertion_loss < 4.0
                and extinction_ratio > 12.0
                and bandwidth > 25.0
            )

            timestamp = datetime(2026, 5, 20) + timedelta(
                days=int(np.random.randint(0, 5)),
                minutes=int(np.random.randint(0, 1440)),
            )

            test_rows.append({
                "test_id": f"TST_{len(test_rows):06d}",
                "device_id": device_id,
                "wafer_id": fab["wafer_id"],
                "design_id": fab["design_id"],
                "test_station": np.random.choice(test_stations),
                "operator": np.random.choice(operators),
                "timestamp": timestamp.isoformat(),
                "temperature_c": round(np.random.normal(22, 1.5), 2),
                "humidity_percent": round(np.random.normal(40, 5), 2),
                "wavelength_nm": round(np.random.normal(design["target_wavelength_nm"], 2), 2),
                "insertion_loss_db": round(insertion_loss, 2),
                "extinction_ratio_db": round(extinction_ratio, 2),
                "bandwidth_ghz": round(bandwidth, 2),
                "raw_sweep_file": f"raw_sweeps/{device_id}_sweep.csv",
                "passed": bool(passed),
            })

    tests = pd.DataFrame(test_rows)

    to_json_records(designs, "designs.json")
    to_json_records(fabrication, "fabrication.json")
    to_json_records(tests, "tests.json")

    print("\nDataset summary:")
    print(f"- Designs: {len(designs)}")
    print(f"- Wafers: {len(fabrication)}")
    print(f"- Device tests: {len(tests)}")
    print(f"- Overall yield: {tests['passed'].mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
