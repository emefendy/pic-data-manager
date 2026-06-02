import shutil
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd


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


def load_joined_results() -> pd.DataFrame:
    conn, tmp_path = open_db_in_tmp()
    df = pd.read_sql_query("SELECT * FROM joined_ai_results", conn)
    conn.close()
    tmp_path.unlink(missing_ok=True)
    return df


def compute_performance_by_foundry(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("foundry")
        .agg(
            avg_insertion_loss_db=("insertion_loss_db", "mean"),
            avg_extinction_ratio_db=("extinction_ratio_db", "mean"),
            avg_bandwidth_ghz=("bandwidth_ghz", "mean"),
            avg_defect_count=("defect_count", "mean"),
            avg_measured_width_nm=("measured_width_nm", "mean"),
            avg_sidewall_roughness_nm=("sidewall_roughness_nm", "mean"),
            yield_rate=("passed", "mean"),
            num_devices=("device_id", "count"),
        )
        .reset_index()
    )

    summary["yield_rate"] = summary["yield_rate"] * 100
    return summary.round(2)


def compute_process_performance_correlations(df: pd.DataFrame) -> pd.DataFrame:
    process_features = [
        "film_thickness_nm",
        "etch_depth_nm",
        "measured_width_nm",
        "width_error_nm",
        "sidewall_roughness_nm",
        "defect_count",
    ]

    performance_metrics = [
        "insertion_loss_db",
        "extinction_ratio_db",
        "bandwidth_ghz",
        "passed",
    ]

    corr_rows = []

    temp = df.copy()
    temp["passed"] = temp["passed"].astype(int)

    for process_feature in process_features:
        for metric in performance_metrics:
            corr = temp[process_feature].corr(temp[metric])

            corr_rows.append({
                "process_feature": process_feature,
                "performance_metric": metric,
                "correlation": corr,
            })

    correlations = pd.DataFrame(corr_rows)

    return correlations.sort_values(
        "correlation",
        key=lambda s: s.abs(),
        ascending=False,
    ).round(3)


def save_analysis_outputs(foundry_summary, correlations):
    conn, tmp_path = open_db_in_tmp()

    foundry_summary.to_sql(
        "foundry_performance_summary",
        conn,
        if_exists="replace",
        index=False,
    )

    correlations.to_sql(
        "process_performance_correlations",
        conn,
        if_exists="replace",
        index=False,
    )

    conn.close()
    sync_db_from_tmp(tmp_path)


def main():
    df = load_joined_results()

    foundry_summary = compute_performance_by_foundry(df)
    correlations = compute_process_performance_correlations(df)

    save_analysis_outputs(foundry_summary, correlations)

    print("Foundry performance summary:")
    print(foundry_summary)

    print("\nStrongest process-performance relationships:")
    print(correlations.head(10))

    print("\nSaved tables:")
    print("- foundry_performance_summary")
    print("- process_performance_correlations")


if __name__ == "__main__":
    main()
