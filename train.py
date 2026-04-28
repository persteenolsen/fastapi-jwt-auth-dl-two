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
        # Simple feedforward neural network
        self.net = nn.Sequential(
            nn.Linear(5, 16),   # Input layer (5 features → 16 hidden units)
            nn.ReLU(),          # Activation function
            nn.Linear(16, 16),  # Hidden layer
            nn.ReLU(),          # Activation function
            nn.Linear(16, 1)    # Output layer (predict single price value)
        )

    def forward(self, x):
        # Forward pass through the network
        return self.net(x)


# -------- TRAIN --------
def train():
    # Generate synthetic dataset
    X, y = generate_data(3000)

    # -------- SPLIT --------
    # Split into training (80%) and validation (20%)
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # -------- INPUT NORMALIZATION --------
    # Compute mean and std from training data only
    X_mean = X_train.mean(dim=0)
    X_std = X_train.std(dim=0)

    # Normalize inputs (important for stable training)
    X_train = (X_train - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std

    # -------- TARGET NORMALIZATION (IMPORTANT FIX) --------
    # Normalize target values to improve learning stability
    y_mean = y_train.mean()
    y_std = y_train.std()

    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm = (y_val - y_mean) / y_std

    # -------- MODEL --------
    model = HousePriceModel().float()

    # Mean Squared Error loss for regression
    criterion = nn.MSELoss()

    # Adam optimizer for parameter updates
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # -------- TRAIN LOOP --------
    for epoch in range(200):
        model.train()  # Set model to training mode

        optimizer.zero_grad()  # Reset gradients

        preds = model(X_train)  # Forward pass
        loss = criterion(preds, y_train_norm)  # Compute loss

        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        # -------- VALIDATION --------
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            val_preds = model(X_val)
            val_loss = criterion(val_preds, y_val_norm)
            rmse = torch.sqrt(val_loss)  # RMSE for interpretability

        # Print progress every 20 epochs
        if epoch % 20 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {loss.item():.4f} | "
                f"Val RMSE (norm): {rmse.item():.4f}"
            )

    # -------- SAVE PREPROCESSING --------
    # Store normalization parameters for use in inference (e.g., FastAPI)
    preprocessing = {
        "x_mean": X_mean.tolist(),
        "x_std": X_std.tolist(),
        "y_mean": y_mean.item(),
        "y_std": y_std.item(),
        "features": ["size", "rooms", "age", "distance", "income_area"]
    }

    # Save preprocessing config to JSON file
    with open("preprocessing.json", "w") as f:
        json.dump(preprocessing, f)

    print("Saved preprocessing.json")

    # -------- EXPORT ONNX --------
    # Export trained model to ONNX format for deployment
    model.eval()

    # Dummy input for tracing model graph
    dummy_input = torch.randn(1, 5, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        "model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},   # Allow variable batch size
            "output": {0: "batch_size"}
        }
    )

    print("Exported model.onnx")


if __name__ == "__main__":
    # Run training when script is executed directly
    train()