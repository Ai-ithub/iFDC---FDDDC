# === config.py ===
from pathlib import Path

# Base project directory (directory where this config file resides)
PROJECT_DIR = Path(__file__).parent

# Directory for synthetic datasets
DATA_PATH = PROJECT_DIR / "dataset" / "synthetic"

# Directory to store trained models
MODEL_PATH = PROJECT_DIR / "models"
