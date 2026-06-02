import shutil
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import shap
from sklearn.ensemble import IsolationForest


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "pic_data.db"


def open_db_in_tmp() -> tuple[sqlite3.Connection, Path]:
    """Copy DB to /tmp and open it there (avoids filesystem locking issues)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    shutil.copy2(DB_PATH, tmp_path)
    return sqlite3.connect(tmp_path), tmp_path


def sync_db_from_tmp(tmp_path: Path):
    """Copy the modified temp DB back to the project folder."""
    shutil.copy2(tmp_path, DB_PATH)
    tmp_path.unlink(missing_ok=True)


def load_joined_data() -> pd.DataFrame:
    conn, tmp_path = open_db_in_tmp()

    query = """
    SELECT
        t.test_id,
        t.device_id,
        t.wafer_id,
        t.design_id,
        d.device_type,
        d.design_version,
        d.waveguide_width_nm,
        d.coupler_gap_nm,
        d.ring_radius_um,
        d.target_wavelength_nm,
        d.simulated_loss_db,
        d.simulated_bandwidth_ghz,
        f.fab_run,
        f.foundry,
        f.process_recipe,
        f.film_thickness_nm,
        f.etch_depth_nm,
        f.measured_width_nm,
        f.width_error_nm,
        f.sidewall_roughness_nm,
        f.defect_count,
        t.test_station,
        t.operator,
        t.timestamp,
        t.temperature_c,
        t.humidity_percent,
        t.wavelength_nm,
        t.insertion_loss_db,
        t.extinction_ratio_db,
        t.bandwidth_ghz,
        t.passed
    FROM tests t
    JOIN designs d
        ON t.design_id = d.design_id
    JOIN fabrication f
        ON t.wafer_id = f.wafer_id
    """

    df = pd.read_sql_query(query, conn)
    conn.close()
    tmp_path.unlink(missing_ok=True)
    return df


ANOMALY_FEATURES = [
    "waveguide_width_nm",
    "coupler_gap_nm",
    "simulated_loss_db",
    "simulated_bandwidth_ghz",
    "film_thickness_nm",
    "etch_depth_nm",
    "measured_width_nm",
    "width_error_nm",
    "sidewall_roughness_nm",
    "defect_count",
    "insertion_loss_db",
    "extinction_ratio_db",
    "bandwidth_ghz",
]


def add_ai_anomaly_flags(df: pd.DataFrame) -> tuple[pd.DataFrame, IsolationForest, pd.DataFrame]:
    df = df.copy()

    model = IsolationForest(
        contamination=0.05,
        random_state=42,
    )

    X = df[ANOMALY_FEATURES]
    df["anomaly_label"] = model.fit_predict(X)
    df["ai_flagged_anomaly"] = df["anomaly_label"] == -1

    # SHAP values 
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap_df = pd.DataFrame(shap_values, columns=ANOMALY_FEATURES)
    shap_df.columns = [f"shap_{c}" for c in shap_df.columns]
    shap_df.insert(0, "device_id", df["device_id"].values)
    shap_df.insert(1, "ai_flagged_anomaly", df["ai_flagged_anomaly"].values)

    return df, model, shap_df


def save_ai_results(df: pd.DataFrame, shap_df: pd.DataFrame):
    conn, tmp_path = open_db_in_tmp()
    df.to_sql("joined_ai_results", conn, if_exists="replace", index=False)
    shap_df.to_sql("shap_values", conn, if_exists="replace", index=False)
    conn.close()
    sync_db_from_tmp(tmp_path)


def main():
    df = load_joined_data()
    df, model, shap_df = add_ai_anomaly_flags(df)
    save_ai_results(df, shap_df)

    print("AI results saved to table: joined_ai_results")
    print("SHAP values saved to table: shap_values")
    print(f"Anomalies flagged: {df['ai_flagged_anomaly'].sum()}")
    print(df[["device_id", "passed", "ai_flagged_anomaly"]].head())


if __name__ == "__main__":
    main()
