from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import logging
from fastapi import FastAPI

logger = logging.getLogger(__name__)

MODEL_PATH = Path("artifacts/final_models/ucva_stepwise_linear_log_v0.joblib")


@dataclass
class ModelArtifact:
    pipeline: Any
    feature_names: list[str]
    model_name: str
    model_version: str
    training_ranges: dict[str, dict[str, float]]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Loading model from %s...", MODEL_PATH)

    if not MODEL_PATH.exists():
        logger.error("Model file not found: %s", MODEL_PATH)
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    artifact = joblib.load(MODEL_PATH)

    app.state.model_artifact = ModelArtifact(
        pipeline=artifact["pipeline"],
        feature_names=artifact["feature_names"],
        model_name=artifact.get("model_name", "unknown_model"),
        model_version=artifact.get("model_version", "unknown_version"),
        training_ranges=artifact.get("training_ranges", {}),
    )

    logger.info(
        "Model '%s' %s loaded successfully", 
        app.state.model_artifact.model_name, 
        app.state.model_artifact.model_version
    )

    yield

    # Shutdown
    logger.info("Shutting down...")
