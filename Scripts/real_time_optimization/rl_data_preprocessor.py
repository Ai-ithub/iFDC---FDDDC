import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# File paths
input_file = '../../dataset/synthetic_fdms_chunks/FDMS_well_WELL_1.parquet'
output_dir = 'RL_preprocessed_data'
output_file = os.path.join(output_dir, 'FDMS_well_WELL_1_preprocessed.parquet')

os.makedirs(output_dir, exist_ok=True)

# Selected columns for training the agent
selected_columns = [
    'Depth_m',
    'ROP_mph', # For reward function
    'MWD_Vibration_g',
    'Formation_Type', # Categorical (needs encoding)
    'Pore_Pressure_psi',
    'Mud_Weight_ppg',
    'Viscosity_cP',
    'Fluid_Loss_Risk',
    'Emulsion_Risk',
    'Formation_Damage_Index', # For reward function
    'WOB_kgf', # Action
    'Torque_Nm', # Action
    'Pump_Pressure_psi', # Action
    'Mud_FlowRate_LPM' # Action
]
# Loading the dataset
df = pd.read_parquet(input_file, columns=selected_columns)

# Removing outliers using IQR method
def get_outlier_mask(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 2 * IQR
    upper = Q3 + 2 * IQR
    return (df[col] >= lower) & (df[col] <= upper)

# Only numeric columns
numeric_cols = [col for col in df.columns if df[col].dtype in [np.float64, np.int64]]
# Build a single combined mask
combined_mask = np.ones(len(df), dtype=bool)
for col in numeric_cols:
    if not col.startswith('Formation_Type'):
        combined_mask &= get_outlier_mask(df, col)

# Applying filter
df = df[combined_mask]

# Handling missing values
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

# Encoding Formation_Type
formation_dummies = pd.get_dummies(df['Formation_Type'], drop_first=True)
formation_dummies = formation_dummies.astype(int)
df = pd.concat([df.drop(columns=['Formation_Type']), formation_dummies], axis=1)

# Normalizing numerical features
numeric_cols = [col for col in df.columns if col not in ['Limestone', 'Sandstone', 'Shale']]
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Adding previous timestep actions (t-1)
for col in ['WOB_kgf', 'Torque_Nm', 'Pump_Pressure_psi', 'Mud_FlowRate_LPM']:
    df[f'{col}_t-1'] = df[col].shift(1)

# Dropping the first row with NaN from shift
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Saving the processed file
df.to_parquet(output_file, index=False, compression="snappy")
