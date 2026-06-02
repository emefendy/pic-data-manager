import json
import shutil
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DB_PATH = ROOT / "pic_data.db"


def load_json(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename

    with open(path, "r") as f:
        data = json.load(f)

    return pd.DataFrame(data)


def validate_columns(df: pd.DataFrame, required_columns: list[str], table_name: str):
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"{table_name} is missing columns: {missing}")


def main():
    designs = load_json("designs.json")
    fabrication = load_json("fabrication.json")
    tests = load_json("tests.json")

    validate_columns(
        designs,
        [
            "design_id",
            "device_type",
            "design_version",
            "layout_file",
            "waveguide_width_nm",
            "coupler_gap_nm",
            "target_wavelength_nm",
            "simulated_loss_db",
        ],
        "designs",
    )

    validate_columns(
        fabrication,
        [
            "wafer_id",
            "fab_run",
            "foundry",
            "design_id",
            "film_thickness_nm",
            "etch_depth_nm",
            "measured_width_nm",
            "sidewall_roughness_nm",
            "defect_count",
            "metrology_file",
        ],
        "fabrication",
    )

    validate_columns(
        tests,
        [
            "test_id",
            "device_id",
            "wafer_id",
            "design_id",
            "test_station",
            "timestamp",
            "wavelength_nm",
            "insertion_loss_db",
            "extinction_ratio_db",
            "bandwidth_ghz",
            "raw_sweep_file",
            "passed",
        ],
        "tests",
    )

    # Build in a temp file first 
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    conn = sqlite3.connect(tmp_path)
    designs.to_sql("designs", conn, if_exists="replace", index=False)
    fabrication.to_sql("fabrication", conn, if_exists="replace", index=False)
    tests.to_sql("tests", conn, if_exists="replace", index=False)
    conn.close()

    shutil.copy2(tmp_path, DB_PATH)
    tmp_path.unlink(missing_ok=True)

    print(f"Database created: {DB_PATH}")
    print("Tables created: designs, fabrication, tests")


if __name__ == "__main__":
    main()
