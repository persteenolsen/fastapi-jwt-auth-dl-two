# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import json

from data import generate_data


# -------- MODEL --------
class HousePriceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)


# -------- TRAIN --------
def train():
    X, y = generate_data(3000)

    # -------- SPLIT --------
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # -------- INPUT NORMALIZATION --------
    X_mean = X_train.mean(dim=0)
    X_std = X_train.std(dim=0)

    X_train = (X_train - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std

    # -------- TARGET NORMALIZATION (IMPORTANT FIX) --------
    y_mean = y_train.mean()
    y_std = y_train.std()

    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm = (y_val - y_mean) / y_std

    # -------- MODEL --------
    model = HousePriceModel().float()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # -------- TRAIN LOOP --------
    for epoch in range(200):
        model.train()

        optimizer.zero_grad()

        preds = model(X_train)
        loss = criterion(preds, y_train_norm)

        loss.backward()
        optimizer.step()

        # -------- VALIDATION --------
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = criterion(val_preds, y_val_norm)
            rmse = torch.sqrt(val_loss)

        if epoch % 20 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {loss.item():.4f} | "
                f"Val RMSE (norm): {rmse.item():.4f}"
            )

    # -------- SAVE PREPROCESSING --------
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

    # -------- EXPORT ONNX --------
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