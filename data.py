# data.py
import torch

def generate_data(n_samples=1000, save_csv=False):
    torch.manual_seed(42)

    size = torch.rand(n_samples, 1) * 200
    rooms = torch.randint(1, 7, (n_samples, 1)).float()
    age = torch.rand(n_samples, 1) * 50
    distance = torch.rand(n_samples, 1) * 30
    income_area = torch.rand(n_samples, 1) * 100

    size_per_room = size / rooms
    location_penalty = distance ** 1.5

    price = (
        size * 2500 +
        rooms * 15000 +
        size_per_room * 500 +
        income_area * 2000 -
        age * 1800 -
        location_penalty * 4000 +
        torch.randn(n_samples, 1) * 20000
    )

    X = torch.cat([size, rooms, age, distance, income_area], dim=1)
    y = price

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
    X, y = generate_data(1000, save_csv=True)
    print("Dataset generated:", X.shape, y.shape)