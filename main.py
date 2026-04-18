import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

HF_SPACE_URL = "https://aalia-laghari-mindease-api.hf.space/gradio_api/queue/join"

class PredictRequest(BaseModel):
    inputs: str

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: PredictRequest):
    # Use the direct call endpoint instead
    response = requests.post(
        "https://aalia-laghari-mindease-api.hf.space/gradio_api/call/predict",
        json={"data": [data.inputs]},
        timeout=60
    )
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    event_id = response.json()["event_id"]

    # Poll for result
    result_response = requests.get(
        f"https://aalia-laghari-mindease-api.hf.space/gradio_api/call/predict/{event_id}",
        timeout=60,
        stream=True
    )
    for line in result_response.iter_lines():
        if line and line.startswith(b"data: "):
            import json
            result = json.loads(line[6:])
            label = result[0]["label"]
            score = result[0]["score"]
            return {
                "stressed": label == "LABEL_1",
                "confidence": round(score * 100, 1),
                "label": label
            }

    raise HTTPException(status_code=500, detail="No result from model")
