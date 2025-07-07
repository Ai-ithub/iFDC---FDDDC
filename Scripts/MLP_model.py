import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import shap
import os

# ------------------ [Configuration] ------------------
DATA_PATH = "FDMS_well_WELL_1.parquet"
MODEL_OUTPUT_PATH = "models/fluid_loss_best_model.onnx"
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
EPOCHS = 30
PATIENCE = 5

# ------------------ [Load Data] ------------------
def load_data(path):
    print("📥 Loading data...")
    df = pd.read_parquet(path)
    df["Fluid_Loss_Label"] = (df["Fluid_Loss_Risk"] > 0.5).astype(int)
    print("✅ Data loaded and ready.")
    return df

# ------------------ [Preprocess Data] ------------------
def preprocess_data(df):
    print("⚙️ Preprocessing data...")
    target = "Fluid_Loss_Label"
    features = [col for col in df.columns if col not in [target, "Fluid_Loss_Risk", "timestamp", "WELL_ID", "LAT", "LONG"]]

    cat_cols = df[features].select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    X = df[features]
    y = df[target]

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    print("✅ Preprocessing completed.")
    return X_train, X_test, y_train, y_test, features

# ------------------ [Define Custom Dataset] ------------------
class FluidLossDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ------------------ [Define Model] ------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )

    def forward(self, x):
        return torch.relu(self.net(x) + x)

class FluidLossModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            ResidualBlock(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

# ------------------ [Train Model] ------------------
def train_model(model, train_loader, test_loader, device):
    print("🚀 Starting model training...")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)

    best_f1 = 0
    best_state = None
    wait = 0

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        y_preds, y_true = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                preds = torch.sigmoid(model(xb)).cpu().numpy()
                y_preds.extend(preds)
                y_true.extend(yb.numpy())

        y_bin = np.array(y_preds) > 0.5
        f1 = f1_score(y_true, y_bin)
        scheduler.step(1 - f1)
        print(f"📉 Epoch {epoch+1}: F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print("🛑 Early stopping.")
                break

    model.load_state_dict(best_state)
    print("✅ Model training completed.")
    return model

# ------------------ [Evaluate Model] ------------------
def evaluate_model(model, X_test, y_test, device):
    print("📊 Starting model evaluation...")
    model.eval()
    with torch.no_grad():
        y_pred_prob = torch.sigmoid(model(torch.tensor(X_test, dtype=torch.float32).to(device))).cpu().numpy()
        y_pred = y_pred_prob > 0.5

    results = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_pred_prob),
    }
    print("✅ Model evaluation completed.")
    return results

# ------------------ [Prediction Function for SHAP] ------------------
def model_predict_fn(model, device):
    def predict(x):
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
            return torch.sigmoid(model(x_tensor)).cpu().numpy()
    return predict

# ------------------ [SHAP Analysis] ------------------
def run_shap(model, X_train, X_test, features, device):
    print("📈 Calculating SHAP values...")
    explainer = shap.KernelExplainer(model_predict_fn(model, device), X_train[:300])
    shap_values = explainer.shap_values(X_test[:150])
    shap.summary_plot(shap_values, features=X_test[:150], feature_names=features)
    print("✅ SHAP analysis completed.")

# ------------------ [Export Model] ------------------
def export_model_onnx(model, input_dim, device, output_path):
    print("💾 Saving model in ONNX format...")
    dummy_input = torch.randn(1, input_dim, dtype=torch.float32).to(device)
    torch.onnx.export(model, dummy_input, MODEL_OUTPUT_PATH , input_names=["input"], output_names=["output"], opset_version=11)
    print(f"✅ Model saved: {output_path}")

# ------------------ [Run All] ------------------
def main():
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, features = preprocess_data(df)

    train_ds = FluidLossDataset(X_train, y_train)
    test_ds = FluidLossDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FluidLossModel(input_dim=X_train.shape[1]).to(device)

    model = train_model(model, train_loader, test_loader, device)

    results = evaluate_model(model, X_test, y_test, device)
    print("\n📊 Final Results:")
    for k, v in results.items():
        print(f"{k}: {v:.5f}")

    run_shap(model, X_train, X_test, features, device)
    export_model_onnx(model, X_train.shape[1], device, MODEL_OUTPUT_PATH)

    print("\n🏁 All steps completed successfully.")

if __name__ == "__main__":
    main()
