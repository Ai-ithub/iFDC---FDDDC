# === src/report_generator.py ===
import pandas as pd
import matplotlib.pyplot as plt
import os


def generate_report():
    """
    Generate a boxplot report comparing RMSE scores across models or wells.

    The function reads the evaluation metrics CSV file, verifies required columns,
    creates a boxplot of RMSE scores grouped by model or well_id, and saves the plot
    to the results/plots directory.
    """
    # Ensure the directory for plots exists
    os.makedirs("results/plots", exist_ok=True)

    # Load evaluation metrics
    df = pd.read_csv("results/metrics.csv")

    # Verify required column is present
    if 'rmse' not in df.columns:
        raise ValueError("Column 'rmse' not found in evaluation results")

    # Determine grouping column, default to 'well_id'
    group_column = 'model' if 'model' in df.columns else 'well_id'

    # Create boxplot for RMSE scores by group
    plt.figure(figsize=(10, 5))
    df.boxplot(column="rmse", by=group_column, grid=False)
    plt.title("Model Performance Comparison by RMSE")
    plt.suptitle("")  # Remove automatic suptitle
    plt.xlabel(group_column.capitalize())
    plt.ylabel("RMSE Score")
    plt.xticks(rotation=45)

    # Save plot to file
    plt.tight_layout()
    plt.savefig("results/plots/rmse_comparison.png")
    plt.close()
