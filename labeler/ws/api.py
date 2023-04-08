from pathlib import Path
from typing import List
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from labeler.evaluator import ClassifierEvaluator
from pydantic import BaseModel


class Input(BaseModel):
    data: List[str]

class Prediction(BaseModel):
    probabilities: List[List[float]]

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)

model = ClassifierEvaluator(directory=Path('./data/trained_model'))

@app.post("/predict", response_model=Prediction)
async def predict(input: Input):
    probabilities = model.infer(input.data)
    return {"probabilities": probabilities}

