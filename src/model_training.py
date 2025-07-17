# === src/model_training.py ===
from sklearn.ensemble import RandomForestRegressor
import joblib
from pathlib import Path
from .data_preparation import load_prepare_data


def train_models():
    """
    Train a Random Forest Regressor model for each well dataset
    and save the trained models to disk.

    Each model is saved as 'models/well_<well_id>.pkl'.
    """
    Path("models").mkdir(exist_ok=True)  # Ensure models directory exists

    for well_id in range(1, 11):
        # Load features and target for the current well
        X, y = load_prepare_data(well_id)

        # Initialize and train Random Forest Regressor
        model = RandomForestRegressor()
        model.fit(X, y)

        # Save the trained model
        joblib.dump(model, f"models/well_{well_id}.pkl")
