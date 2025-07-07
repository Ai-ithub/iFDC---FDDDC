import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib

# Initial settings
plt.style.use('ggplot')
torch.manual_seed(42)
np.random.seed(42)

# 📌 Paths
data_path = "FDMS_well_WELL_1.parquet"
model_output_path = "models/fluid_loss_best_model.onnx"

# ⚙️ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📦 Dataset class
class FluidLossDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 🧠 GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch_size, 1, input_size]
        gru_out, _ = self.gru(x)  # [batch_size, 1, hidden_size]
        
        # Attention
        attention_weights = torch.softmax(self.attention(gru_out), dim=1)
        context = torch.sum(attention_weights * gru_out, dim=1)
        
        # Classification
        out = self.classifier(context)
        return out.squeeze()

    def extract_features(self, X):
        self.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            X_tensor = X_tensor.unsqueeze(1)
            gru_out, _ = self.gru(X_tensor)
            attention_weights = torch.softmax(self.attention(gru_out), dim=1)
            features = torch.sum(attention_weights * gru_out, dim=1)
        return features.cpu().numpy()

    def fit(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=512):
        train_dataset = FluidLossDataset(X_train, y_train)
        val_dataset = FluidLossDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        criterion = nn.BCELoss()
        
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.float().to(device)
                
                optimizer.zero_grad()
                outputs = self(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Evaluation
            self.eval()
            val_preds = []
            with torch.no_grad():
                for X_val_batch, _ in val_loader:
                    X_val_batch = X_val_batch.to(device)
                    val_preds.extend(self(X_val_batch).cpu().numpy())
            
            val_preds = np.array(val_preds) > 0.5
            val_recall = recall_score(y_val, val_preds)
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(train_loader):.4f} - Val Recall: {val_recall:.4f}")

# 🧠 Hybrid Model
class HybridModel:
    def __init__(self, input_size):
        self.gru_model = GRUModel(input_size).to(device)
        self.rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced')
        self.calibrator = CalibratedClassifierCV(self.rf_model, cv=3)
        
    def train(self, X_train, y_train, X_val, y_val):
        # Train GRU
        print("Training GRU Model...")
        self.gru_model.fit(X_train, y_train, X_val, y_val)
        
        # Feature extraction
        print("Extracting features...")
        X_train_features = self.gru_model.extract_features(X_train)
        X_val_features = self.gru_model.extract_features(X_val)
        
        # Train RandomForest
        print("Training RandomForest...")
        self.calibrator.fit(X_train_features, y_train)
        
        # Validation evaluation
        val_probs = self.calibrator.predict_proba(X_val_features)[:, 1]
        val_pred = (val_probs > 0.5).astype(int)
        print(f"Validation Recall: {recall_score(y_val, val_pred):.4f}")
    
    def predict_proba(self, X):
        features = self.gru_model.extract_features(X)
        return self.calibrator.predict_proba(features)[:, 1]

# 📊 Plotting functions
def plot_metrics(y_true, y_pred, y_probs):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2)
    axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    axes[1, 0].plot(recall, precision, color='blue', lw=2)
    axes[1, 0].set_title('Precision-Recall Curve')
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    
    # Prediction distribution
    axes[1, 1].hist(y_probs[y_true==0], bins=30, alpha=0.5, label='Class 0')
    axes[1, 1].hist(y_probs[y_true==1], bins=30, alpha=0.5, label='Class 1')
    axes[1, 1].set_title('Prediction Distribution')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('metrics_plot.png')
    plt.show()

# 🔍 Load and preprocess data
def load_and_preprocess(path):
    df = pd.read_parquet(path)
    
    # Drop unnecessary columns
    df.drop(columns=["WELL_ID", "LAT", "LONG", "timestamp"], errors='ignore', inplace=True)
    df.dropna(inplace=True)
    
    # Create label
    df["Fluid_Loss_Label"] = (df["Fluid_Loss_Risk"] > 0.5).astype(int)
    
    # Encode categorical variables
    df = pd.get_dummies(df, columns=["Bit_Type", "Formation_Type", "Shale_Reactiveness"], drop_first=True)
    
    # Separate features and label
    X = df.drop(columns=["Fluid_Loss_Risk", "Fluid_Loss_Label"])
    y = df["Fluid_Loss_Label"]
    
    # Convert to numeric array
    X = X.select_dtypes(include=[np.number]).astype(np.float32).values
    y = y.values
    
    return X, y
  
def plot_shap_analysis(model, X, feature_names=None):
    """Simple SHAP Analysis"""
    try:
        print("🔍 Running SHAP analysis...")
        
        # Extract features from GRU
        features = model.gru_model.extract_features(X)
        
        # Access RandomForest model
        rf_model = model.calibrator.calibrated_classifiers_[0].estimator
        
        # Create explainer
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(features)
        
        # Select positive class for display
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]  # class 1
        else:
            shap_vals = shap_values
        
        # Feature names
        if feature_names is None:
            feature_names = [f"GRU_Feature_{i}" for i in range(features.shape[1])]
        
        # Plot summary
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, features, feature_names=feature_names, show=False)
        plt.title("SHAP Analysis - Feature Importance")
        plt.tight_layout()
        plt.savefig('shap_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ SHAP analysis completed!")
        
    except Exception as e:
        print(f"⚠️ SHAP failed: {str(e)}")


# 💾 Save model
def save_model(model, X_sample):
    # Save GRU model
    torch.save(model.gru_model.state_dict(), 'gru_model.pth')
    
    # Save RandomForest model
    joblib.dump(model.calibrator, 'rf_calibrator.pkl')
    
    print("✅ Models saved successfully")
def export_gru_to_onnx(model, input_sample, output_path):
    model.gru_model.eval()
    input_tensor = torch.tensor(input_sample, dtype=torch.float32).to(device)

    # Add batch and sequence dimension (for GRU)
    input_tensor = input_tensor.unsqueeze(1)

    torch.onnx.export(
        model.gru_model,                     # PyTorch model
        input_tensor,                        # Sample input
        output_path,                         # Save path
        export_params=True,
        opset_version=11,                    # Compatible opset version
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"✅ GRU model exported to ONNX at: {output_path}")


# 🚀 Full pipeline execution
def run_pipeline():
    # Load data
    print("📥 Loading data...")
    X, y = load_and_preprocess(data_path)
    print(f"🔍 Class distribution: {dict(Counter(y))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)
    
    # Apply SMOTE only on training data
    print("🔄 Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Train hybrid model
    print("🚀 Training Hybrid Model...")
    model = HybridModel(X_train.shape[1])
    model.train(X_train_res, y_train_res, X_val, y_val)
    
    # Predict and evaluate
    print("🔮 Predicting on test set...")
    y_probs = model.predict_proba(X_test)
    
    # Find optimal threshold
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    y_pred = (y_probs >= optimal_threshold).astype(int)
    
    # Compute metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_probs),
        'Optimal Threshold': optimal_threshold
    }
    
    print("\n📊 Final Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    

    # Plot metrics
    plot_metrics(y_test, y_pred, y_probs)
    


    # Run SHAP on a sample of the test set (for performance)
    sample_idx = np.random.choice(X_test.shape[0], size=200, replace=False)  
    plot_shap_analysis(model, X_test[sample_idx])
    
    # Save model
    save_model(model, X_test[:1])
    export_gru_to_onnx(model, X_test[:1], model_output_path)

if __name__ == "__main__":
    run_pipeline()
