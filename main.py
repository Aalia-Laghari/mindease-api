import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

HF_SPACE_URL = "https://aalia-laghari-mindease-api.hf.space/run/predict"

class PredictRequest(BaseModel):
    inputs: str

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: PredictRequest):
    response = requests.post(
        HF_SPACE_URL,
        json={"data": [data.inputs]},
        timeout=60
    )
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()
