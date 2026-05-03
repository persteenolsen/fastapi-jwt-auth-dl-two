import torch
import torch.nn as nn
import torch.optim as optim
import json

from data import generate_data


# -------- MODEL --------
class HousePriceModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Small neural network for regression
        self.net = nn.Sequential(
            nn.Linear(5, 6),   # 5 inputs → 6 hidden units (kept small for smooth output)
            nn.ReLU(),         # non-linearity (allows patterns but still stable)
            nn.Linear(6, 1)    # output: predicted house price
        )

    def forward(self, x):
        return self.net(x)


# -------- TRAIN --------
def train():

    # Generate synthetic dataset (features + price)
    X, y = generate_data(3000)

    # -------- TRAIN / VALIDATION SPLIT --------
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # -------- INPUT NORMALIZATION --------
    # Normalize features for stable training
    X_mean = X_train.mean(dim=0)
    X_std = X_train.std(dim=0) + 1e-8   # avoid division by zero

    X_train = (X_train - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std

    # -------- TARGET NORMALIZATION --------
    # Normalize prices so model trains more easily
    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8

    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm = (y_val - y_mean) / y_std

    # -------- MODEL --------
    model = HousePriceModel().float()

    # Loss function for regression
    criterion = nn.MSELoss()

    # Optimizer with weight decay = smoother predictions (reduces overfitting)
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)

    # -------- EARLY STOPPING SETUP --------
    best_val = float("inf")
    patience = 40
    wait = 0

    # -------- TRAIN LOOP --------
    for epoch in range(350):

        model.train()

        # forward pass
        preds = model(X_train)
        loss = criterion(preds, y_train_norm)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # -------- VALIDATION --------
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = criterion(val_preds, y_val_norm)
            rmse = torch.sqrt(val_loss)

        # Early stopping logic (stop if no improvement)
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        # Progress log
        if epoch % 20 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {loss.item():.4f} | "
                f"Val RMSE: {rmse.item():.4f}"
            )

    # -------- SAVE NORMALIZATION --------
    preprocessing = {
        "x_mean": X_mean.tolist(),
        "x_std": X_std.tolist(),
        "y_mean": y_mean.item(),
        "y_std": y_std.item(),
        "features": ["size", "rooms", "age", "distance", "income_area"]
    }

    with open("preprocessing.json", "w") as f:
        json.dump(preprocessing, f)

    print("Saved preprocessing.json")

    # -------- EXPORT MODEL (ONNX) --------
    model.eval()

    dummy_input = torch.randn(1, 5, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        "model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )

    print("Exported model.onnx")


if __name__ == "__main__":
    train()