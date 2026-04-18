from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

app = FastAPI(title="MindEase API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_MODEL = "Aalia-Laghari/mindease-stress-model"

sensor_model = pickle.load(open("stress_trained.sav", "rb"))
scaler = pickle.load(open("scaler.sav", "rb"))
text_tokenizer = DistilBertTokenizerFast.from_pretrained(HF_MODEL)
text_model = DistilBertForSequenceClassification.from_pretrained(
    HF_MODEL,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
text_model.eval()


class StressRequest(BaseModel):
    q1: str
    q2: str
    q3: str
    q4: str
    q5: str
    humidity: float
    temperature: float
    steps: int


class StressResponse(BaseModel):
    level: str
    score: float
    description: str
    quote: str
    music_url: str


quotes = {
    "High Stress": [
        "Even the darkest night will end and the sun will rise. — Victor Hugo",
        "Stress is not what happens to us. It's our response to what happens. — Hans Selye",
    ],
    "Medium Stress": [
        "Don't be pushed by your problems, be led by your dreams. — Ralph Waldo Emerson",
        "Take rest; a field that has rested gives a bountiful crop. — Ovid",
    ],
    "Low Stress": [
        "Peace comes from within. Do not seek it without. — Buddha",
        "Happiness is not something ready-made. It comes from your own actions. — Dalai Lama",
    ],
}

music_links = {
    "High Stress": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
    "Medium Stress": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-3.mp3",
    "Low Stress": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3",
}

descriptions = {
    "High Stress": "Your stress levels are elevated. It's okay — let's take it one breath at a time.",
    "Medium Stress": "You're managing, but there's some tension. A little rest will go a long way.",
    "Low Stress": "You're calm and balanced. Keep nurturing that inner peace.",
}


def predict_sensor(humidity, temperature, steps):
    data = np.asarray([humidity, temperature, steps]).reshape(1, -1)
    data = scaler.transform(data)
    return sensor_model.predict(data)[0] / 2.0


def predict_text(text):
    inputs = text_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = text_model(**inputs)
        pred_class = torch.argmax(outputs.logits, dim=1).item()
    return pred_class / 2.0


@app.get("/")
def root():
    return {"status": "MindEase API is running"}


@app.post("/predict", response_model=StressResponse)
def predict(req: StressRequest):
    user_text = " ".join([req.q1, req.q2, req.q3, req.q4, req.q5])
    text_score = predict_text(user_text)
    sensor_score = predict_sensor(req.humidity, req.temperature, req.steps)
    final_score = (text_score + sensor_score) / 2

    if final_score > 0.6:
        level = "High Stress"
    elif final_score > 0.3:
        level = "Medium Stress"
    else:
        level = "Low Stress"

    return StressResponse(
        level=level,
        score=round(final_score, 3),
        description=descriptions[level],
        quote=np.random.choice(quotes[level]),
        music_url=music_links[level],
    )
