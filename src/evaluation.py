# === src/evaluation.py ===
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from .data_preparation import load_prepare_data
import joblib
from pathlib import Path


def evaluate_all_wells():
    """
    Evaluate trained models across all wells using RMSE and R^2 metrics.

    Loads each model and dataset, performs prediction, computes metrics,
    and saves the results to a CSV file.

    Returns:
        results (list of dict): Evaluation metrics for each well
    """
    results = []

    for well_id in range(1, 11):
        # Load data and corresponding model
        X, y = load_prepare_data(well_id)
        model = joblib.load(Path("models") / f"well_{well_id}.pkl")

        # Predict and compute metrics
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        rmse = mse ** 0.5  # Root Mean Squared Error
        r2 = r2_score(y, y_pred)

        results.append({
            "well_id": well_id,
            "rmse": rmse,
            "r2": r2
        })

    # Save evaluation results
    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv("results/metrics.csv", index=False)

    return results
