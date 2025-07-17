# === src/data_preparation.py ===
import pandas as pd
import numpy as np
from pathlib import Path
from config import DATA_PATH


def load_prepare_data(well_id: int):
    """
    Load and preprocess data for a given well.

    Args:
        well_id (int): The ID of the well to load.

    Returns:
        X (DataFrame): Feature matrix after preprocessing
        y (Series): Target variable (Formation Damage Index)
    """
    file_path = Path(DATA_PATH) / \
        f"synthetic_fdms_chunks/FDMS_well_WELL_{well_id}.parquet"
    df = pd.read_parquet(file_path)

    # Drop unused columns and apply one-hot encoding to categorical variables
    X = pd.get_dummies(
        df.drop(["Formation_Damage_Index", "WELL_ID", "timestamp"], axis=1)
    )
    y = df["Formation_Damage_Index"]

    return X, y
