from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "reallabdata_raw"
_FALLBACK = Path.home() / "Downloads" / "reallabdata"
_LEGACY_FALLBACK = Path.home() / "Downloads" / "ScreeningDatasetV2_candidate"


def resolve_dataset(dataset: str | os.PathLike | None = None) -> Path:
    if dataset:
        p = Path(dataset).expanduser()
    elif os.environ.get("SCREENING_DATASET"):
        p = Path(os.environ["SCREENING_DATASET"]).expanduser()
    elif DEFAULT_DATASET.exists():
        p = DEFAULT_DATASET
    elif _FALLBACK.exists():
        p = _FALLBACK
    else:
        p = _LEGACY_FALLBACK
    if not p.exists():
        raise FileNotFoundError(
            f"Raw lab dataset not found at {p}. Put it at data/reallabdata_raw/, "
            "pass --dataset PATH, or set SCREENING_DATASET."
        )
    return p
