# main.py
import os
import json
import numpy as np
import onnxruntime as ort
from datetime import datetime, timedelta

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from dotenv import load_dotenv
from jose import jwt, JWTError, ExpiredSignatureError

# -------- LOAD ENV --------
load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET", "dev_secret")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))

# ✅ NEW: credentials from .env
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

# -----------------------------
# INIT APP
# -----------------------------
# app = FastAPI()
app = FastAPI(
    title="FastAPI + JWT + Deep Learning + House Price Prediction (v5)",
    description="27-04-2026 - FastAPI + JWT + Deep Learning + House Price Prediction Neural Network trained by PyTorch and exported to ONNX",
    version="5.0.0",
    contact={
        "name": "Per Olsen",
        "url": "https://persteenolsen.netlify.app",
    },
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# -------- TOKEN --------
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# -------- LOGIN --------
@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # ✅ uses .env values instead of hardcoded dict
    if (
        form_data.username != ADMIN_USERNAME
        or form_data.password != ADMIN_PASSWORD
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )

    token = create_access_token({"sub": form_data.username})

    return {
        "access_token": token,
        "token_type": "bearer"
    }

# -------- LOAD MODEL --------
session = ort.InferenceSession("model.onnx")

# -------- LOAD PREPROCESSING --------
with open("preprocessing.json", "r") as f:
    norm = json.load(f)

X_mean = np.array(norm["x_mean"], dtype=np.float32)
X_std = np.array(norm["x_std"], dtype=np.float32)

y_mean = float(norm["y_mean"])
y_std = float(norm["y_std"])

# -------- REST OF YOUR CODE (UNCHANGED) --------
class HouseFeatures(BaseModel):
    size: float
    rooms: float
    age: float
    distance: float
    income_area: float

@app.get("/")
def root():
    return {"message": "House Price API v5 + PyTorch + ONNX"}

@app.post("/predict")
def predict(data: HouseFeatures, user=Depends(get_current_user)):
    x = np.array([[
        data.size,
        data.rooms,
        data.age,
        data.distance,
        data.income_area
    ]], dtype=np.float32)

    x = (x - X_mean) / X_std

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: x})

    pred_norm = float(output[0][0][0])
    price = pred_norm * y_std + y_mean

    return {
        "predicted_price": round(price, 2),
        "user": user
    }