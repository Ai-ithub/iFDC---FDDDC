import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Fixed Parameters
NUM_WELLS = 10
SAMPLES_PER_WELL = 1000  # Reduced from 15M for practicality
CHUNK_SIZE = 1000
OUTPUT_DIR = "synthetic_fdms_chunks"

# Constants
BIT_TYPES = ["PDC", "Tricone", "Diamond"]
FORMATION_TYPES = ["Sandstone", "Limestone", "Shale", "Dolomite"]
SHALE_REACTIVITY = ["Low", "Medium", "High"]

# Risk Calculation Functions


def fluid_loss_risk(viscosity, solid_content):
    return np.clip((viscosity / 120) * (solid_content / 20), 0, 1)


def emulsion_risk(oil_water_ratio):
    """Simplified emulsion risk calculation"""
    return np.clip((100 - oil_water_ratio) / 100, 0, 1)


def reactivity_score(shale_reactiveness):
    return np.select(
        [shale_reactiveness == "High", shale_reactiveness == "Medium"],
        [1.0, 0.5],
        default=0.2
    )

# Data Generation Functions


def generate_well_data(well_id, num_samples):
    shift = well_id * 0.1
    scale = 1 + (well_id % 5) * 0.05

    depth = np.random.normal(3000 + shift*500, 800 *
                             scale, num_samples).clip(1000, 6000)
    mud_weight = np.random.normal(
        11 + shift, 1.5 * scale, num_samples).clip(8.5, 15)

    return {
        "Depth_m": depth,
        "ROP_mph": np.random.normal(20 + shift*2, 8 * scale, num_samples).clip(5, 50),
        "WOB_kgf": np.random.normal(15000 + shift*1000, 5000 * scale, num_samples).clip(5000, 30000),
        "Torque_Nm": np.random.normal(1000 + shift*50, 400 * scale, num_samples).clip(200, 2000),
        "Mud_Weight_ppg": mud_weight,
        "Viscosity_cP": np.random.normal(70 + shift*5, 20 * scale, num_samples).clip(30, 120),
        "Solid_Content_%": np.random.uniform(1, 20, num_samples),
        "Oil_Water_Ratio": np.random.uniform(10, 90, num_samples),
        # Removed Emulsion_Stability as it wasn't being generated
        "Shale_Reactiveness": np.random.choice(SHALE_REACTIVITY, num_samples),
        "Formation_Type": np.random.choice(FORMATION_TYPES, num_samples),
        "Bit_Type": np.random.choice(BIT_TYPES, num_samples),
    }


# Main Execution
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start_time = datetime.now()

    for well_id in range(1, NUM_WELLS + 1):
        well_path = os.path.join(
            OUTPUT_DIR, f"FDMS_well_WELL_{well_id}.parquet")

        if os.path.exists(well_path):
            os.remove(well_path)

        # Generate data in chunks
        for chunk_start in range(0, SAMPLES_PER_WELL, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, SAMPLES_PER_WELL)
            chunk_size = chunk_end - chunk_start

            # Generate base data
            data = generate_well_data(well_id, chunk_size)
            df = pd.DataFrame(data)

            # Calculate risks
            df["Fluid_Loss_Risk"] = fluid_loss_risk(
                df["Viscosity_cP"], df["Solid_Content_%"])
            df["Emulsion_Risk"] = emulsion_risk(df["Oil_Water_Ratio"])
            df["Rock_Fluid_Reactivity"] = reactivity_score(
                df["Shale_Reactiveness"])

            # Calculate Damage Index
            df["Formation_Damage_Index"] = (
                0.4 * df["Fluid_Loss_Risk"] +
                0.3 * df["Rock_Fluid_Reactivity"] +
                0.2 * df["Emulsion_Risk"] +
                0.1 * np.random.normal(0, 0.05, chunk_size)
            ).clip(0, 1)

            # Add metadata
            df["WELL_ID"] = f"WELL_{well_id}"
            df["timestamp"] = datetime.now(
            ) + pd.to_timedelta(np.arange(chunk_size), unit='s')

            # Save to parquet
            df.to_parquet(
                well_path,
                engine='fastparquet',
                compression='snappy',
                append=os.path.exists(well_path)
            )

        print(f"Generated WELL_{well_id} with {SAMPLES_PER_WELL:,} samples")

    print(
        f"\n✅ Completed in {(datetime.now() - start_time).total_seconds():.2f} seconds")
