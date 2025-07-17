# === main.py ===
from src.model_training import train_models
from src.evaluation import evaluate_all_wells
from src.report_generator import generate_report

if __name__ == "__main__":
    # Perform model training (FR-3.1 and FR-3.3 requirements)
    train_models()

    # Evaluate models across all wells (FR-3.2 requirement)
    evaluate_all_wells()

    # Generate final evaluation report
    generate_report()
