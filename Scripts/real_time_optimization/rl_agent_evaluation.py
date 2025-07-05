import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from d3rlpy.dataset import MDPDataset
import d3rlpy
from offline_rl_agent import DATA_PATH, STATE_COLS, ACTION_COLS, MODEL_SAVE_PATH
from sklearn.model_selection import train_test_split


df = pd.read_parquet(DATA_PATH)
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
timeouts_test[-1] = False

# Testing dataset
test_dataset = MDPDataset(
    observations=s_test,
    actions=a_test,
    rewards=r_test,
    terminals=d_test,
    timeouts=timeouts_test
)

# Loading the model
cql = d3rlpy.algos.CQLConfig().create()
cql.build_with_dataset(test_dataset)
cql.load_model(MODEL_SAVE_PATH)
pred_actions = cql.predict(s_test)

# Evaluations
# Computing MSE between predicted actions and real actions
mse = np.mean((pred_actions - a_test) ** 2)
print(f"Mean Squared Error between predicted actions and test actions: {mse:.4f}")

# Computing reward
def compute_reward(states, actions):
    ROP = states[:, STATE_COLS.index('ROP_mph')]
    Damage = states[:, STATE_COLS.index('Formation_Damage_Index')]
    return 2.0 * ROP - 3.0 * Damage

predicted_rewards = compute_reward(s_test, pred_actions)
actual_rewards = compute_reward(s_test, a_test)
print(f"Mean predicted reward: {np.mean(predicted_rewards):.3f}")
print(f"Mean actual reward: {np.mean(actual_rewards):.3f}")

# Action Comparison Plots
plt.figure(figsize=(12, 8))
for i, col in enumerate(ACTION_COLS):
    plt.subplot(2, 2, i+1)
    plt.plot(a_test[:, i], label='Dataset Action')
    plt.plot(pred_actions[:, i], label='Agent Action')
    plt.title(f'Action: {col}')
    plt.legend()
plt.tight_layout()
plt.savefig('output_results/action_comparison.png')

# Plotting the effects of each feature in the reward function on the agent's decisions
damage = s_test[:, STATE_COLS.index('Formation_Damage_Index')]
median_damage = np.median(damage)
low_damage_idx = damage <= median_damage
high_damage_idx = damage > median_damage
for i, action_name in enumerate(ACTION_COLS):
    plt.figure(figsize=(6,4))
    plt.hist(pred_actions[low_damage_idx, i], bins=30, alpha=0.7, label='Low damage states')
    plt.hist(pred_actions[high_damage_idx, i], bins=30, alpha=0.7, label='High damage states')
    plt.title(f'Predicted action {action_name} distribution by damage level')
    plt.xlabel('Action value')
    plt.ylabel('Count')
    plt.legend()
plt.savefig('output_results/Pred_actions_by_damage.png')

rop_idx = STATE_COLS.index('ROP_mph')
rop = s_test[:, rop_idx]
median_rop = np.median(rop)
low_rop_idx = rop <= median_rop
high_rop_idx = rop > median_rop
for i, action_name in enumerate(ACTION_COLS):
    plt.figure(figsize=(6,4))
    plt.hist(pred_actions[low_rop_idx, i], bins=30, alpha=0.7, label='Low ROP states')
    plt.hist(pred_actions[high_rop_idx, i], bins=30, alpha=0.7, label='High ROP states')
    plt.title(f'Predicted action {action_name} distribution by ROP level')
    plt.xlabel('Action value')
    plt.ylabel('Count')
    plt.legend()
plt.savefig('output_results/Pred_actions_by_ROP.png')
