import os
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset


# Input features per timestep (no acceleration)
INPUT_COLS: List[str] = ["Spatial_Gap", "Speed_LV", "Speed_FAV"]
TARGET_COL = "Acc_FAV"
ID_COL = "ID_FAV"
TIME_COL = "Time_Index"
TRAJ_COL = "Trajectory_ID"


def load_fav_sequences(
    data_dir: str = None,
    fav_id: int = 1,
    seq_len: int = 10,
    csv_file: Optional[str] = None,
) -> TensorDataset:
    """
    Read all CSVs under data_dir, keep rows with ID_FAV == fav_id, and build
    fixed-length sequences for sequence-to-one LSTM training.

    Features per timestep: Spatial_Gap, Speed_LV, Speed_FAV.
    Target: next-frame Acc_FAV following each input window.

    Returns:
        TensorDataset with X shape (N, seq_len, 3) and y shape (N, 1).
    """
    if csv_file:
        csv_files = [Path(csv_file)]
    else:
        if data_dir is None:
            # Default to project_root/data relative to this file.
            data_dir = Path(__file__).resolve().parents[2] / "data"
        else:
            data_dir = Path(data_dir)
        csv_files = sorted(data_dir.glob("*.csv"))
    X_sequences: List[np.ndarray] = []
    y_targets: List[float] = []

    for csv_path in csv_files:
        df = pd.read_csv(
            csv_path,
            usecols=[TRAJ_COL, TIME_COL, ID_COL, *INPUT_COLS, TARGET_COL],
        )
        df = df[df[ID_COL] == fav_id]
        if df.empty:
            continue

        # Build sequences within each trajectory to avoid crossing gaps.
        for _, group in df.groupby(TRAJ_COL):
            group = group.sort_values(TIME_COL)
            inputs = group[INPUT_COLS].to_numpy(dtype=np.float32)   # (T, 3)
            targets = group[TARGET_COL].to_numpy(dtype=np.float32)  # (T,)
            if len(inputs) <= seq_len:
                continue

            for start in range(0, len(inputs) - seq_len):
                end = start + seq_len
                X_sequences.append(inputs[start:end])
                # Next-frame acceleration of FAV as target.
                y_targets.append(targets[end])

    if not X_sequences:
        # Return empty tensors to keep the signature consistent.
        return TensorDataset(
            torch.empty(0, seq_len, len(INPUT_COLS)),
            torch.empty(0, 1),
        )

    X = torch.from_numpy(np.stack(X_sequences)).float()
    y = torch.from_numpy(np.array(y_targets, dtype=np.float32)).unsqueeze(-1)
    return TensorDataset(X, y)


__all__ = ["load_fav_sequences"]
