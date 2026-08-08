"""Reusable inference pipeline for the KDD Cup autoencoder."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from keras.models import load_model


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"


@lru_cache(maxsize=1)
def load_artifacts() -> tuple[Any, Any, Any, dict[str, Any]]:
    """Load and cache the trained model and preprocessing artifacts."""
    model = load_model(MODEL_DIR / "autoencoder.keras", compile=False)
    encoder = joblib.load(MODEL_DIR / "encoder.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text())
    return model, encoder, scaler, metadata


def expected_columns() -> list[str]:
    """Return the raw input columns expected by the deployed pipeline."""
    _, _, _, metadata = load_artifacts()
    return metadata["num_cols"] + metadata["cat_cols"]


def categorical_options() -> dict[str, list[str]]:
    """Return the categorical values learned by the fitted encoder."""
    _, encoder, _, metadata = load_artifacts()
    return {
        column: [str(value) for value in categories]
        for column, categories in zip(metadata["cat_cols"], encoder.categories_)
    }


def preprocess_connections(connections: pd.DataFrame) -> np.ndarray:
    """Reproduce the notebook's encode-then-scale feature pipeline."""
    model, encoder, scaler, metadata = load_artifacts()
    required = metadata["num_cols"] + metadata["cat_cols"]
    missing = [column for column in required if column not in connections.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    frame = connections.loc[:, required].copy()
    for column in metadata["num_cols"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    for column in metadata["cat_cols"]:
        frame[column] = frame[column].astype(str)

    numeric = frame[metadata["num_cols"]].to_numpy(dtype=np.float64)
    categorical = encoder.transform(frame[metadata["cat_cols"]])
    if hasattr(categorical, "toarray"):
        categorical = categorical.toarray()

    unscaled_features = np.hstack([numeric, categorical])
    features = scaler.transform(unscaled_features).astype(np.float32)
    expected_width = int(model.input_shape[-1])
    if features.shape[1] != expected_width:
        raise ValueError(
            f"Preprocessing produced {features.shape[1]} features; "
            f"the model expects {expected_width}."
        )
    return features


def predict_connections(connections: pd.DataFrame) -> pd.DataFrame:
    """Predict NORMAL or ATTACK for one or more raw KDD connections."""
    model, _, _, metadata = load_artifacts()
    features = preprocess_connections(connections)
    reconstructions = model.predict(features, verbose=0)
    scores = np.mean(np.square(features - reconstructions), axis=1)
    threshold = float(metadata["threshold"])
    predictions = np.where(scores > threshold, "ATTACK", "NORMAL")

    return pd.DataFrame(
        {
            "prediction": predictions,
            "anomaly_score": scores,
            "threshold": threshold,
            "score_ratio": scores / threshold,
        },
        index=connections.index,
    )


def predict_connection(connection: Mapping[str, Any]) -> dict[str, Any]:
    """Predict a single raw connection and return JSON-compatible values."""
    result = predict_connections(pd.DataFrame([dict(connection)])).iloc[0]
    return {
        "prediction": str(result["prediction"]),
        "anomaly_score": float(result["anomaly_score"]),
        "threshold": float(result["threshold"]),
        "score_ratio": float(result["score_ratio"]),
    }
