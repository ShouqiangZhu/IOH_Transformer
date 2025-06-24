# model/utils.py
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score

def load_data(csv_path):
    """Load and preprocess time-series input data."""
    df = pd.read_csv(csv_path)
    features = df.drop(columns=["label", "patient_id"]).values
    labels = df["label"].values
    features = features.reshape(-1, 10, 8)  # [samples, time steps, features]
    split_idx = int(0.8 * len(features))
    return (
        torch.tensor(features[:split_idx], dtype=torch.float32),
        torch.tensor(labels[:split_idx], dtype=torch.float32),
        torch.tensor(features[split_idx:], dtype=torch.float32),
        torch.tensor(labels[split_idx:], dtype=torch.float32),
    )

def compute_metrics(y_true, y_probs):
    """Calculate and display model metrics."""
    auc = roc_auc_score(y_true, y_probs)
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    f1 = f1_score(y_true, (np.array(y_probs) > 0.5).astype(int))
    print(f"AUC: {auc:.4f} | F1 Score: {f1:.4f} | Recall @ 0.5: {np.mean(recall):.4f}")
