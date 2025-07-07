# FDMS Data Generation Code Documentation (Formation Damage Monitoring System)

---

## Introduction

This code is designed to generate a simulated dataset from oil well drilling data. The goal is to produce data with characteristics close to real drilling data for use in geological analysis, risk modeling, and machine learning.

---

## General Overview

- Data is generated for **10 wells** with different geographical specifications.
- For each well, approximately **15,552,000 records (samples)** are generated, each representing a one-second interval.
- The data includes numerical parameters, categorical features, various risks, and combined indices.
- Data is saved as `Parquet` files.
- The data covers a **6-month recording period**.

---

## Data Columns and Descriptions


| Column Name                  | Description                                                                  |
|------------------------------|------------------------------------------------------------------------------|
| `Depth_m`                    | Measured depth in meters                                                     |
| `ROP_mph`                    | Rate of Penetration in meters per hour                                       |
| `WOB_kgf`                    | Weight on Bit in kilogram-force                                              |
| `Torque_Nm`                  | Applied torque in Newton-meters                                              |
| `Pump_Pressure_psi`          | Pump pressure in pounds per square inch                                      |
| `Mud_FlowRate_LPM`           | Mud flow rate in liters per minute                                           |
| `MWD_Vibration_g`            | Vibration measured while drilling, in g-force                                |
| `Bit_Type`                   | Type of drill bit (`PDC`, `Tricone`, `Diamond`)                              |
| `Mud_Weight_ppg`             | Drilling mud weight in pounds per gallon                                     |
| `Viscosity_cP`               | Mud viscosity in centipoise                                                  |
| `Plastic_Viscosity`          | Plastic viscosity, derived from overall viscosity                            |
| `Yield_Point`                | Yield point of the fluid, related to shear stress                            |
| `pH_Level`                   | Acidity/alkalinity level of the drilling fluid                               |
| `Solid_Content_%`            | Percentage of solid particles in the mud                                     |
| `Chloride_Concentration_mgL` | Concentration of chloride in the mud, milligrams per liter                   |
| `Oil_Water_Ratio`            | Ratio of oil to water in the mud (0–100 scale)                               |
| `Emulsion_Stability`         | Stability of the emulsion on a scale from 0 to 100                           |
| `Formation_Type`             | Type of rock formation (`Shale`, `Sandstone`, etc.)                          |
| `Pore_Pressure_psi`          | Estimated pore pressure in the formation, in psi                             |
| `Fracture_Gradient_ppg`      | Pressure gradient at which fractures may occur, in ppg                       |
| `Stress_Tensor_MPa`          | In-situ stress value in megapascals                                          |
| `Young_Modulus_GPa`          | Elastic modulus (Young’s modulus) of the rock in GPa                         |
| `Poisson_Ratio`              | Poisson’s ratio of the rock (unitless, typically 0.2–0.35)                   |
| `Brittleness_Index`          | Brittleness index of the rock (0 to 1 scale)                                 |
| `Shale_Reactiveness`         | Rock-fluid reactivity level (`Low`, `Medium`, `High`)                        |
| `Fluid_Loss_Risk`            | Risk index for fluid loss (0 to 1, based on viscosity and solids)            |
| `Emulsion_Risk`              | Risk index for emulsion instability (0 to 1)                                 |
| `Rock_Fluid_Reactivity`      | Numeric reactivity score (0 = Low, 0.5 = Medium, 1 = High)                   |
| `Formation_Damage_Index`     | Composite index indicating potential for formation damage (0 to ~1)          |
| `WELL_ID`                    | Unique identifier for each well (e.g. `WELL_1`)                              |
| `LAT`                        | Latitude of the well                                                         |
| `LONG`                       | Longitude of the well                                                        |
| `timestamp`                  | Timestamp of the recorded measurement (1-second resolution from Jan 1, 2023) |

## Data Generation Process

1. **Define well information:**  
   Specify coordinates and identifiers for 10 different wells.

2. **Define initial parameters:**  
   Set constants and initial parameters, including rock types, bit types, mud weight range, and fracture gradient.

3. **Generate numerical data:**  
   Use normal or uniform distributions to generate depth, mud weight, fracture gradient, solids content, etc.

4. **Assign categorical values:**  
   Randomly select bit type, rock type, and shale reactivity.

5. **Calculate risks:**  
   - **Fluid Loss Risk** based on mud weight and fracture gradient.  
   - **Emulsion Risk** based on oil-water ratio and solids content.  
   - **Rock-Fluid Reactivity** for interaction risks.

6. **Create combined formation damage index:**  
   Combine risk scores and other parameters numerically to calculate the **Formation Damage Index**.

7. **Add outlier data:**  
   Approximately 0.2% of the data is set with extreme high or low values to simulate anomalies.

8. **Add missing values:**  
   Randomly remove values in some numerical columns to simulate missing data.

9. **Add well specifications:**  
   Each record includes the well identifier and geographical coordinates.

10. **Save the data:**  
    Save each well's DataFrame as a `Parquet` file in the specified path.

---

## Example Code for Generating a Sample Dataset

```python
import numpy as np
import pandas as pd

# Generate drilling depth using a normal distribution
depth = np.random.normal(loc=1500, scale=200, size=num_rows_per_well)

# Generate mud weight uniformly between 9.5 and 15 ppg
mud_weight = np.random.uniform(9.5, 15, size=num_rows_per_well)

# Randomly select bit type
bit_type = np.random.choice(["PDC", "Roller Cone", "Diamond"], size=num_rows_per_well)

# Function to calculate fluid loss risk
def fluid_loss_risk(row):
    if row['Mud_Weight_ppg'] > row['Fracture_Gradient_psi']:
        return "High"
    elif abs(row['Mud_Weight_ppg'] - row['Fracture_Gradient_psi']) < 1:
        return "Medium"
    else:
        return "Low"

df = pd.DataFrame({
    "Depth_m": depth,
    "Mud_Weight_ppg": mud_weight,
    "Bit_Type": bit_type,
    # Add other columns similarly
})

df["Fluid_Loss_Risk"] = df.apply(fluid_loss_risk, axis=1)
