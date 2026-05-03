import json
import numpy as np
import onnxruntime as ort


# -------- LOAD PREPROCESSING --------
with open("preprocessing.json", "r") as f:
    prep = json.load(f)

x_mean = np.array(prep["x_mean"], dtype=np.float32)
x_std = np.array(prep["x_std"], dtype=np.float32)
y_mean = prep["y_mean"]
y_std = prep["y_std"]

# -------- LOAD MODEL --------
session = ort.InferenceSession("model.onnx")


def predict(features):
    x = np.array(features, dtype=np.float32)

    # normalize input
    x_norm = (x - x_mean) / x_std
    x_norm = x_norm.reshape(1, -1)

    # run ONNX model
    pred_norm = session.run(None, {"input": x_norm})[0][0][0]

    # denormalize output
    price = pred_norm * y_std + y_mean
    return float(price)


# -------- BASE HOUSE --------
# [size, rooms, age, distance, income_area]
base = [100, 3, 10, 5, 50]


print("\n--- AGE TEST (price should decrease as age increases) ---")
for age in [0, 5, 10, 20, 40, 80]:
    features = base.copy()
    features[2] = age
    price = predict(features)
    print(f"Age: {age:3d} -> Price: {price:,.2f}")


print("\n--- SIZE TEST (price should increase as size increases) ---")
for size in [50, 75, 100, 150, 200, 300]:
    features = base.copy()
    features[0] = size
    price = predict(features)
    print(f"Size: {size:3d} -> Price: {price:,.2f}")