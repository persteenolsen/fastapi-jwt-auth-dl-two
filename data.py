# data.py
import torch

def generate_data(n_samples=1000, save_csv=False):
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Generate synthetic feature data
    size = torch.rand(n_samples, 1) * 200              # House size in m² (0–200)
    rooms = torch.randint(1, 7, (n_samples, 1)).float()  # Number of rooms (1–6)
    age = torch.rand(n_samples, 1) * 50               # Age of the house (0–50 years)
    distance = torch.rand(n_samples, 1) * 30          # Distance to city center (0–30 km)
    income_area = torch.rand(n_samples, 1) * 100      # Area income index (0–100)

    # Derived features for more realistic relationships
    size_per_room = size / rooms                      # Average room size
    location_penalty = distance ** 1.5                # Nonlinear penalty for distance

    # Simulated price calculation with noise
    price = (
        size * 2500 +                # Larger houses cost more
        rooms * 15000 +              # More rooms increase price
        size_per_room * 500 +        # Larger rooms add value
        income_area * 2000 -         # Wealthier area increases price
        age * 1800 -                 # Older houses decrease value
        location_penalty * 4000 +    # Farther distance reduces price
        torch.randn(n_samples, 1) * 20000  # Add random noise
    )

    # Combine features into input tensor
    X = torch.cat([size, rooms, age, distance, income_area], dim=1)
    y = price  # Target variable

    # Optionally save dataset to CSV
    if save_csv:
        import pandas as pd
        df = pd.DataFrame(X.numpy(), columns=[
            "size", "rooms", "age", "distance", "income_area"
        ])
        df["price"] = y.numpy()
        df.to_csv("housing_data.csv", index=False)

    return X, y


# Optional test run
if __name__ == "__main__":
    # Generate dataset and save to file for inspection
    X, y = generate_data(1000, save_csv=True)
    print("Dataset generated:", X.shape, y.shape)