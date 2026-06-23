import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

import fasttext
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import (
    EMBEDDING_DIM,
    HIDDEN_DIM,
    OUTPUT_DIM,
    N_LAYERS,
    DROPOUT,
    DEVICE,
    FASTTEXT_MODEL_PATH,
    MODEL_SAVE_PATH,
)
from src.models.gru import GRUClassifier
from src.pipeline.feature_pipeline import FeaturePipeline
from src.models.predictor import SentencePredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    sentence: str


class PredictResponse(BaseModel):
    sentence: str
    prediction: str
    prediction_numeric: int
    confidence: float
    features: Dict[str, Any]
    tokens: list[str]
    pos_tags: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


predictor: SentencePredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    logger.info("Loading FastText model...")
    embedding_model = fasttext.load_model(FASTTEXT_MODEL_PATH)

    logger.info("Loading saved model checkpoint...")
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)

    feature_columns = checkpoint.get("feature_columns")
    if feature_columns is None:
        from src.config import FEATURE_COLUMNS
        feature_columns = FEATURE_COLUMNS

    scaler = checkpoint.get("scaler", None)
    if scaler is None:
        logger.warning("No scaler found in checkpoint.")

    gru_model = GRUClassifier(EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT).to(DEVICE)
    gru_model.load_state_dict(checkpoint["model_state_dict"])
    gru_model.eval()

    logger.info("Initializing feature pipeline...")
    feature_pipeline = FeaturePipeline(embedding_model=embedding_model, lazy_pos=True)

    logger.info("Creating predictor...")
    predictor = SentencePredictor(
        feature_pipeline_instance=feature_pipeline,
        trained_model=gru_model,
        scaler=scaler,
        feature_columns=feature_columns,
        model_type="gru",
    )
    yield


app = FastAPI(title="Khmer Grammar Checker API", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", model_loaded=predictor is not None)


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    sentence = request.sentence.strip()
    if not sentence:
        raise HTTPException(status_code=400, detail="Sentence cannot be empty")
    try:
        result = predictor.predict_sentence(sentence)
        return PredictResponse(
            sentence=result["sentence"],
            prediction=result["prediction"],
            prediction_numeric=result["prediction_numeric"],
            confidence=result["confidence"],
            features=result["features"],
            tokens=result["tokens"],
            pos_tags=result["pos_tags"],
        )
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
