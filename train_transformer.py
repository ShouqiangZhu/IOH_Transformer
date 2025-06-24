# train/train_transformer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model.transformer_model import TransformerVitalSigns
from model.utils import compute_metrics, load_data

# Load preprocessed tensors
X_train, y_train, X_val, y_val = load_data("data/preprocessed_data.csv")

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# Initialize model
model = TransformerVitalSigns(input_dim=8).cuda()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_val_loss = float('inf')
patience = 3
epochs_no_improve = 0

for epoch in range(50):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.cuda(), y_batch.cuda().float()
        optimizer.zero_grad()
        y_pred = model(X_batch).squeeze()
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    val_loss = 0
    y_true, y_probs = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.cuda(), y_batch.cuda().float()
            y_pred = model(X_batch).squeeze()
            val_loss += criterion(y_pred, y_batch).item()
            y_probs.extend(y_pred.cpu().numpy())
            y_true.extend(y_batch.cpu().numpy())

    print(f"Epoch {epoch+1}: Val Loss = {val_loss:.4f}")
    compute_metrics(y_true, y_probs)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "model_best.pth")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping.")
            break
