# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 17:08:50 2025

@author: hosein
"""


# %% read data

import pandas as pd


import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

file_path = r"C:\Users\hosein\synthetic_fdms_chunks/FDMS_well_WELL_1.parquet"
df = pd.read_parquet(file_path)
df = df.dropna()

# %% describe df



# General information
null_count = df.isnull().sum().sum() 
df_shape = df.shape
print(f'Total number of nulls: {null_count}', f'\ndf shapes: {df_shape}')
print("General DataFrame Information:")
print(df.info())
print("\nDescriptive Statistics:")
print(df.describe())
print("\nNumber of null values in each column:")
print(df.isnull().sum())
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())
print("\nUnique values in categorical columns:")
for col in df.select_dtypes(include='object').columns:
    print(f"{col}: {df[col].nunique()} unique values")
    
    
    
#%% RF 

df = df.drop('timestamp', axis=1)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming the dataframe is already loaded
# Drop the timestamp column

# Encode categorical columns
categorical_cols = ['Bit_Type', 'Formation_Type', 'Shale_Reactiveness', 'WELL_ID']
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Define features (X) and target (y)
X = df.drop('Fluid_Loss_Risk', axis=1)  # Features
y = df['Fluid_Loss_Risk']  # Target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=100,  # Number of trees
    max_depth=6,       # Maximum depth of trees
    random_state=42
)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics in simple terms
print("Model Performance Results:")
print(f"Mean Absolute Error (MAE): On average, predictions are off by {mae:.2f} units from the actual values.")
print(f"Mean Absolute Percentage Error (MAPE): Predictions differ from actual values by {mape:.2f}% on average.")
print(f"R² Score: The model explains {r2*100:.2f}% of the variation in the data.")

# Plot 1: Scatter plot of actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Ideal line
plt.xlabel('Actual Values (Fluid Loss Risk)')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.grid(True)
plt.show()

# Plot 2: Distribution of prediction errors
errors = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(errors, bins=30, kde=True, color='purple')
plt.xlabel('Prediction Error (Actual - Predicted)')
plt.title('Distribution of Prediction Errors')
plt.grid(True)
plt.show()

# Feature importance
feature_importance = rf_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importance (Factors influencing predictions):")
print(importance_df)

# Plot 3: Feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance in Predicting Fluid Loss Risk')
plt.xlabel('Importance')

plt.show()



# %% gba    
from sklearn.ensemble import GradientBoostingRegressor  # Changed to Gradient Boosting
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming the dataframe is already loaded
# Drop the timestamp column
#df = df.drop('timestamp', axis=1)

# Encode categorical columns
categorical_cols = ['Bit_Type', 'Formation_Type', 'Shale_Reactiveness', 'WELL_ID']
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Define features (X) and target (y)
X = df.drop('Fluid_Loss_Risk', axis=1)  # Features
y = df['Fluid_Loss_Risk']  # Target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
gb_model = GradientBoostingRegressor(
    n_estimators=100,  # Number of boosting stages (trees)
    max_depth=6,       # Maximum depth of trees
    learning_rate=0.1, # Step size for updates
    random_state=42
)

# Train the model
gb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gb_model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics in simple terms
print("Model Performance Results:")
print(f"Mean Absolute Error (MAE): On average, predictions are off by {mae:.2f} units from the actual values.")
print(f"Mean Absolute Percentage Error (MAPE): Predictions differ from actual values by {mape:.2f}% on average.")
print(f"R² Score: The model explains {r2*100:.2f}% of the variation in the data.")

# Plot 1: Scatter plot of actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Ideal line
plt.xlabel('Actual Values (Fluid Loss Risk)')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.grid(True)
plt.show()

# Plot 2: Distribution of prediction errors
errors = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(errors, bins=30, kde=True, color='purple')
plt.xlabel('Prediction Error (Actual - Predicted)')
plt.title('Distribution of Prediction Errors')
plt.grid(True)
plt.show()

# Feature importance
feature_importance = gb_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importance (Factors influencing predictions):")
print(importance_df)

# Plot 3: Feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance in Predicting Fluid Loss Risk')
plt.xlabel('Importance')

plt.show()

# %%% train Model: xgboost

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error  # For regression evaluation



# Drop the timestamp column (XGBoost cannot handle datetime directly)
df = df.drop('timestamp', axis=1)

# Encode categorical columns (Bit_Type, Formation_Type, Shale_Reactiveness, WELL_ID)
categorical_cols = ['Bit_Type', 'Formation_Type', 'Shale_Reactiveness', 'WELL_ID']
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Define features (X) and target (y)
# Assuming 'Fluid_Loss_Risk' is the -- target variable --
X = df.drop('Fluid_Loss_Risk', axis=1)  # Features
y = df['Fluid_Loss_Risk']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost model
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',  # For regression; use 'binary:logistic' for classification
    n_estimators=100,              # Number of trees
    learning_rate=0.1,             # Step size
    max_depth=6,                   # Maximum depth of trees
    random_state=42
)

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = xgb_model.predict(X_test)

# Evaluate the model (for regression, using Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse}")

# Optional: Feature importance
feature_importance = xgb_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importance:")
print(importance_df)
    