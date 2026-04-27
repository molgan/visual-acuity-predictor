import logging

from fastapi import Depends, FastAPI

from app.dependencies import get_model_artifact
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    BatchPredictionItemResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.services.model_loader import ModelArtifact, lifespan
from app.services.predict import make_prediction, make_batch_prediction

logging.basicConfig(level=logging.INFO)


app = FastAPI(
    title="Visual Acuity Estimation API",
    version="1.0.0",
    description="Demo API for estimating preoperative visual acuity",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"message": "API works"}


@app.get("/health")
def health(artifact: ModelArtifact = Depends(get_model_artifact)):
    return {
        "status": "ok",
        "model_loaded": artifact is not None,
    }


@app.get("/model-info")
def model_info(artifact: ModelArtifact = Depends(get_model_artifact)):
    return {
        "model_name": artifact.model_name,
        "model_version": artifact.model_version,
        "features": artifact.feature_names,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    artifact: ModelArtifact = Depends(get_model_artifact),
):
    prediction, imputed_fields, warnings = make_prediction(request, artifact)

    return PredictionResponse(
        prediction=prediction,
        model_name=artifact.model_name,
        model_version=artifact.model_version,
        imputed_fields=imputed_fields,
        warnings=warnings,
    )


@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(
    request: BatchPredictionRequest,
    artifact: ModelArtifact = Depends(get_model_artifact),
):
    results = make_batch_prediction(request.items, artifact)

    return BatchPredictionResponse(
        model_name=artifact.model_name,
        model_version=artifact.model_version,
        predictions=[
            BatchPredictionItemResponse(**item)
            for item in results
        ],
    )