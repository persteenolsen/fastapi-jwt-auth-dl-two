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
# Load environment variables from .env file
load_dotenv()

# JWT configuration (with defaults for development)
JWT_SECRET = os.getenv("JWT_SECRET", "dev_secret")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))

# Admin credentials loaded from .env
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

# -----------------------------
# INIT APP
# -----------------------------
# Create FastAPI application with metadata (used in Swagger docs)
app = FastAPI(
    title="FastAPI + JWT + Deep Learning + House Price Prediction (v5)",
    description="03-05-2026 - FastAPI + JWT + Deep Learning + House Price Prediction Neural Network trained by PyTorch and exported to ONNX",
    version="5.0.0",
    contact={
        "name": "Per Olsen",
        "url": "https://persteenolsen.netlify.app",
    },
)

# OAuth2 scheme (used to extract Bearer token from requests)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# -------- TOKEN --------
def create_access_token(data: dict):
    # Create a copy of input data to avoid mutation
    to_encode = data.copy()

    # Set token expiration time
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})

    # Encode JWT token
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        # Decode and validate JWT token
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except ExpiredSignatureError:
        # Token is valid but expired
        raise HTTPException(status_code=401, detail="Token expired")
    except JWTError:
        # Token is invalid (tampered, wrong signature, etc.)
        raise HTTPException(status_code=401, detail="Invalid token")

# -------- LOGIN --------
@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Validate credentials against values from .env
    if (
        form_data.username != ADMIN_USERNAME
        or form_data.password != ADMIN_PASSWORD
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )

    # Create JWT token with username as subject
    token = create_access_token({"sub": form_data.username})

    return {
        "access_token": token,
        "token_type": "bearer"
    }

# -------- LOAD MODEL --------
# Load ONNX model for inference (exported from PyTorch)
session = ort.InferenceSession("model.onnx")

# -------- LOAD PREPROCESSING --------
# Load normalization parameters used during training
with open("preprocessing.json", "r") as f:
    norm = json.load(f)

# Convert normalization values to numpy for inference
X_mean = np.array(norm["x_mean"], dtype=np.float32)
X_std = np.array(norm["x_std"], dtype=np.float32)

y_mean = float(norm["y_mean"])
y_std = float(norm["y_std"])

# -------- DATA MODEL --------
# Request body schema for prediction endpoint
class HouseFeatures(BaseModel):
    size: float
    rooms: float
    age: float
    distance: float
    income_area: float

# -------- ROOT ENDPOINT --------
# Simple health/info endpoint
@app.get("/")
def root():
    return {"message": "House Price API v5 + PyTorch + ONNX"}

# -------- PREDICTION --------
@app.post("/predict")
def predict(data: HouseFeatures, user=Depends(get_current_user)):
    # Convert input data into numpy array (shape: [1, 5])
    x = np.array([[
        data.size,
        data.rooms,
        data.age,
        data.distance,
        data.income_area
    ]], dtype=np.float32)

    # Apply same normalization as during training
    x = (x - X_mean) / X_std

    # Get model input name dynamically (ONNX requirement)
    input_name = session.get_inputs()[0].name

    # Run inference
    output = session.run(None, {input_name: x})

    # Extract normalized prediction
    pred_norm = float(output[0][0][0])

    # Convert prediction back to original price scale
    price = pred_norm * y_std + y_mean

    # Return rounded prediction and user info from token
    return {
        "predicted_price": round(price, 2),
        "user": user
    }