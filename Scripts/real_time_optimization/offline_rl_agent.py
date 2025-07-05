import pandas as pd
import numpy as np
import torch
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import CQL, CQLConfig
from sklearn.model_selection import train_test_split

DATA_PATH = 'RL_preprocessed_data/FDMS_well_WELL_1_preprocessed.parquet'
STATE_COLS = [
    'Depth_m', 'ROP_mph', 'MWD_Vibration_g', 'Pore_Pressure_psi',
    'Mud_Weight_ppg', 'Viscosity_cP', 'Fluid_Loss_Risk', 'Emulsion_Risk',
    'Formation_Damage_Index', 'Limestone', 'Sandstone', 'Shale',
    'WOB_kgf_t-1', 'Torque_Nm_t-1', 'Pump_Pressure_psi_t-1', 'Mud_FlowRate_LPM_t-1'
]
ACTION_COLS = ['WOB_kgf', 'Torque_Nm', 'Pump_Pressure_psi', 'Mud_FlowRate_LPM']
MODEL_SAVE_PATH = '../../models/RTO_CQL_agent/trained_cql_model.pt'

# Loading the dataset
df = pd.read_parquet(DATA_PATH)

# Computing rewards
rewards = df['ROP_mph'] * 2.0 - df['Formation_Damage_Index'] * 3.0

# Preparing dataset
states = df[STATE_COLS].values
actions = df[ACTION_COLS].values
rewards = rewards.values

# Computing next_states
next_states = np.roll(states, -1, axis=0)
dones = np.zeros(len(states), dtype=bool)
dones[-1] = True

# Preparing timeouts
timeouts = np.zeros(len(states), dtype=bool)
timeouts[-1] = True

# Splitting
split_indices = np.arange(len(states))
train_idx, test_idx = train_test_split(split_indices, test_size=0.2, random_state=42, shuffle=False)
s_train, s_test = states[train_idx], states[test_idx]
a_train, a_test = actions[train_idx], actions[test_idx]
r_train, r_test = rewards[train_idx], rewards[test_idx]
d_train, d_test = dones[train_idx], dones[test_idx]
timeouts_train, timeouts_test = timeouts[train_idx], timeouts[test_idx]

# Ensuring at least one True in timeouts_train (necessary for CQL)
if np.sum(timeouts_train) == 0:
    timeouts_train[-1] = True

# Final MDPDataset creation
dataset = MDPDataset(
    observations=s_train,
    actions=a_train,
    rewards=r_train,
    terminals=d_train,
    timeouts=timeouts_train
)
# Initializing CQL
config = CQLConfig()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cql = CQL(config=config, device=device, enable_ddp=False)

# Training
print('=============== Started Training The Model ===============')
cql.fit(dataset, n_steps=100000)
print('=============== Finished Training The Model ===============')

# Saving the model
cql.save_model(MODEL_SAVE_PATH)
