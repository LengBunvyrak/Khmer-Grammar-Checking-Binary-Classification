import sys
import torch
import fasttext
import pandas as pd

from src.config import (
    EMBEDDING_DIM,
    HIDDEN_DIM,
    OUTPUT_DIM,
    N_LAYERS,
    DROPOUT,
    DEVICE,
    FASTTEXT_MODEL_PATH,
    MODEL_SAVE_PATH,
    USE_FEATURE_FUSION,
    NUM_EXTRA_FEATURES,
)
from src.models.gru import ImprovedGRUClassifier
from src.pipeline.feature_pipeline import FeaturePipeline
from src.models.predictor import SentencePredictor


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <sentence>")
        sys.exit(1)

    sentence = sys.argv[1]

    print("Loading FastText model...")
    embedding_model = fasttext.load_model(FASTTEXT_MODEL_PATH)

    print("Loading saved model checkpoint...")
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)

    if "feature_columns" in checkpoint:
        feature_columns = checkpoint["feature_columns"]
    else:
        from src.config import FEATURE_COLUMNS
        feature_columns = FEATURE_COLUMNS

    scaler = checkpoint.get("scaler", None)
    if scaler is None:
        print("Warning: No scaler found in checkpoint.")

    gru_model = ImprovedGRUClassifier(
        EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT,
        num_extra_features=NUM_EXTRA_FEATURES, use_feature_fusion=USE_FEATURE_FUSION,
    ).to(DEVICE)
    gru_model.load_state_dict(checkpoint["model_state_dict"])
    gru_model.eval()

    print("Initializing feature pipeline...")
    feature_pipeline = FeaturePipeline(embedding_model=embedding_model, lazy_pos=True)

    print("Creating predictor...")
    predictor = SentencePredictor(
        feature_pipeline_instance=feature_pipeline,
        trained_model=gru_model,
        scaler=scaler,
        feature_columns=feature_columns,
        model_type="gru",
    )

    print(f"\nPredicting for: {sentence}")
    result = predictor.predict_sentence(sentence)

    print(f"Prediction: {result['prediction']} (numeric: {result['prediction_numeric']})")
    if result["confidence"] is not None:
        print(f"Confidence: {result['confidence']:.4f}")

    print("\nExtracted Features:")
    for k, v in result["features"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\nTokens: {result['tokens']}")
    print(f"POS Tags: {result['pos_tags']}")


if __name__ == "__main__":
    main()
