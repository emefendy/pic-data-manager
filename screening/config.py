"""Dataset path resolution.

The raw ScreeningDatasetV2 (~44 MB, 100 captures) is kept *outside* the project
so the repo stays light. Point the pipeline at it via, in priority order:

    1. an explicit ``dataset`` argument / ``--dataset`` CLI flag,
    2. the ``SCREENING_DATASET`` environment variable,
    3. the default download location.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATASET = Path.home() / "Downloads" / "ScreeningDatasetV2_candidate"


def resolve_dataset(dataset: str | os.PathLike | None = None) -> Path:
    if dataset:
        p = Path(dataset).expanduser()
    elif os.environ.get("SCREENING_DATASET"):
        p = Path(os.environ["SCREENING_DATASET"]).expanduser()
    else:
        p = DEFAULT_DATASET
    if not p.exists():
        raise FileNotFoundError(
            f"ScreeningDatasetV2 not found at {p}. Pass --dataset PATH or set "
            "SCREENING_DATASET to the dataset folder."
        )
    return p
