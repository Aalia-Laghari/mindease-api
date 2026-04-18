import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

HF_API_URL = "https://api-inference.huggingface.co/models/Aalia-Laghari/mindease-stress-model"
HF_TOKEN = os.environ.get("HF_TOKEN")  # set this in Railway environment variables

class PredictRequest(BaseModel):
    inputs: str  # adjust type to match what your HF model expects (str or dict)

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: PredictRequest):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(
        HF_API_URL,
        headers=headers,
        json=data.dict(),
        timeout=30
    )
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()
