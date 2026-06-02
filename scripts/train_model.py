import shutil
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "pic_data.db"

FEATURES = [
    "defect_count",
    "sidewall_roughness_nm",
    "width_error_nm",
    "film_thickness_nm",
    "etch_depth_nm",
    "measured_width_nm",
]
TARGET = "insertion_loss_db"


def open_db_in_tmp() -> tuple[sqlite3.Connection, Path]:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    shutil.copy2(DB_PATH, tmp_path)
    return sqlite3.connect(tmp_path), tmp_path


def sync_db_from_tmp(tmp_path: Path):
    shutil.copy2(tmp_path, DB_PATH)
    tmp_path.unlink(missing_ok=True)


def load_data() -> pd.DataFrame:
    conn, tmp_path = open_db_in_tmp()
    df = pd.read_sql_query("SELECT * FROM joined_ai_results", conn)
    conn.close()
    tmp_path.unlink(missing_ok=True)
    return df


class InsertionLossNet(nn.Module):
    """
    Simple feedforward network:
      6 fab features → 16 hidden → 8 hidden → 1 output (insertion loss in dB)
    """
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train(df: pd.DataFrame) -> tuple[np.ndarray, "InsertionLossNet", np.ndarray, np.ndarray]:
    X_raw = df[FEATURES].values.astype(np.float32)
    y_raw = df[TARGET].values.astype(np.float32).reshape(-1, 1)

    X_mean = X_raw.mean(axis=0)
    X_std  = X_raw.std(axis=0) + 1e-8
    X_norm = (X_raw - X_mean) / X_std

    X_all_t = torch.FloatTensor(X_norm.tolist())
    y_all_t = torch.FloatTensor(y_raw.tolist())

    model     = InsertionLossNet(len(FEATURES))
    loss_fn   = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(500):
        model.train()
        pred = model(X_all_t)
        loss = loss_fn(pred, y_all_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch + 1:>3}/500  loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        y_pred = np.array(model(X_all_t).tolist()).flatten()

    return y_pred, model, X_mean, X_std

def save_predictions(df: pd.DataFrame, y_pred: np.ndarray):
    results = df[["device_id", "wafer_id", "design_id", "foundry", TARGET]].copy()
    results["predicted_insertion_loss_db"] = y_pred.round(3)
    results["residual_db"] = (
        results[TARGET] - results["predicted_insertion_loss_db"]
    ).round(3)

    conn, tmp_path = open_db_in_tmp()
    results.to_sql("model_predictions", conn, if_exists="replace", index=False)
    conn.close()
    sync_db_from_tmp(tmp_path)

    return results


def save_model(model: InsertionLossNet, X_mean: np.ndarray, X_std: np.ndarray):
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), models_dir / "insertion_loss_net.pt")
    np.savez(models_dir / "norm_params.npz", X_mean=X_mean, X_std=X_std)
    print(f"Saved model to {models_dir / 'insertion_loss_net.pt'}")
    print(f"Saved norm params to {models_dir / 'norm_params.npz'}")


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    print("Loading data from joined_ai_results...")
    df = load_data()
    print(f"  {len(df)} devices  |  features: {FEATURES}")

    print("\nTraining InsertionLossNet...")
    y_pred, model, X_mean, X_std = train(df)

    results = save_predictions(df, y_pred)
    save_model(model, X_mean, X_std)

    mae  = np.abs(results["residual_db"]).mean()
    rmse = np.sqrt((results["residual_db"] ** 2).mean())

    print(f"\nFull-dataset performance:")
    print(f"  MAE:  {mae:.4f} dB")
    print(f"  RMSE: {rmse:.4f} dB")
    print("\nSaved table: model_predictions")


if __name__ == "__main__":
    main()
