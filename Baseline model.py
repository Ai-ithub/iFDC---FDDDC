import pandas as pd
import numpy as np
import glob
import os
import joblib

import xgboost as xgb
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, LSTM
from tensorflow.keras.callbacks import EarlyStopping


# load the data
data_dir = "synthetic_fdms_chunks"
all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".parquet")]
data = pd.concat([pd.read_parquet(file) for file in all_files], ignore_index=True)

print("✅ All wells loaded:", data["WELL_ID"].nunique(), "wells")


# preprocessing for REGRESSION & XGBOOST
data = data.dropna(subset=["Fluid_Loss_Risk"])
data = data.dropna()

features = ["Mud_Weight_ppg", "Fracture_Gradient_ppg"]
x = data[features]
y = data['Fluid_Loss_Risk']

# standardization
scaler_reg = StandardScaler()
x_scaled = scaler_reg.fit_transform(x)

# train-test split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# train linear regression
reg_model = LinearRegression()
reg_model.fit(x_train, y_train)

# reg evaluation
reg_y_pred = reg_model.predict(x_test)

reg_mse = mean_squared_error(y_test, reg_y_pred)
reg_rmse = np.sqrt(reg_mse)
reg_mae = mean_absolute_error(y_test, reg_y_pred)
reg_r2 = r2_score(y_test, reg_y_pred)

print(f"📊 Evaluation Results for Linear Regression on Fluid_Loss_Risk:")
print(f"  RMSE: {reg_rmse:.4f}")
print(f"  MAE : {reg_mae:.4f}")
print(f"  R²  : {reg_r2:.4f}")

# save linear regression model and scaler
os.makedirs("models", exist_ok=True)
joblib.dump(reg_model, "models/linear_fluid.pkl")
joblib.dump(scaler_reg, "models/scaler_fluid.pkl")

print("✅ Linear Regression model and scaler saved successfully.")

# train xgboost
xg_model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    objective='reg:squarederror'
)
xg_model.fit(x_train, y_train)

# train xgboost
xg_model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    objective='reg:squarederror'
)
xg_model.fit(x_train, y_train)

# save xgboost model
joblib.dump(xg_model, "models/xgboost_fluid_loss_model.pkl")

print("✅ XGBoost model saved successfully.")


# preprocessing for GRU & LSTM
df = data[["timestamp", "Oil_Water_Ratio", "Solid_Content_%", "Emulsion_Risk"]]
df.dropna(inplace=True)
df.sort_values("timestamp", inplace=True)

features = ['Oil_Water_Ratio', 'Solid_Content_%']
target = 'Emulsion_Risk'

# normalization
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features + [target]])

# sequence data
def create_sequences(data, time_steps=60):
    x_t, y_t = [], []
    for i in range(time_steps, len(data)):
        x_t.append(data[i - time_steps:i, :-1]) 
        y_t.append(data[i, -1])                  
    return np.array(x_t), np.array(y_t)

x_t, y_t = create_sequences(scaled, time_steps=60)

# Split
split = int(0.8 * len(x_t))
x_t_train, x_t_test = x_t[:split],x_t[split:]
y_t_train, y_t_test = y_t[:split], y_t[split:]

x_t_train.shape
y_t_train.shape

# GRU

# gru model
gru_model = Sequential([
    GRU(64, return_sequences=False, input_shape=(x_t_train.shape[1], x_t_train.shape[2])),
    Dense(1)
])

gru_model.compile(optimizer='adam', loss='mse')

# train
gru_model.fit(x_t_train, y_t_train,
              epochs=10,
              batch_size=64,
              validation_split=0.1,
              callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
              verbose=1)

# Predict and evaluate
y_pred_gru = gru_model.predict(x_t_test)

print("\nGRU Evaluation:")
print("RMSE:", np.sqrt(mean_squared_error(y_t_test, y_pred_gru)))
print("MAE:", mean_absolute_error(y_t_test, y_pred_gru))
print("R²:", r2_score(y_t_test, y_pred_gru))

# Save lstm model
joblib.dump(gru_model, "lstm_emulsion_model.h5")

print("✅ GRU model saved successfully.")


# LSTM

# LSTM model
lstm_model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(x_t_train.shape[1], x_t_train.shape[2])),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')

# Train
lstm_model.fit(x_t_train, y_t_train,
               epochs=10,
               batch_size=64,
               validation_split=0.1,
               callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
               verbose=1)

# Predict and evaluate
y_pred_lstm = lstm_model.predict(x_t_test)

print("\nLSTM Evaluation:")
print("RMSE:", np.sqrt(mean_squared_error(y_t_test, y_pred_lstm)))
print("MAE:", mean_absolute_error(y_t_test, y_pred_lstm))
print("R²:", r2_score(y_t_test, y_pred_lstm))

# Save lstm model
joblib.dump(lstm_model, "lstm_emulsion_model.h5")

print("✅ LSTM model saved successfully.")