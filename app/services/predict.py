import logging

import pandas as pd
from fastapi import HTTPException

from app.schemas import PredictionRequest
from app.services.model_loader import ModelArtifact

logger = logging.getLogger(__name__)

PLAUSIBLE_MIN = 0.0
PLAUSIBLE_MAX = 1.2


def _get_missing_fields(input_dict: dict) -> list[str]:
    return [k for k, v in input_dict.items() if v is None]


def _validate_not_all_missing(input_dict: dict, missing_fields: list[str]) -> None:
    if len(missing_fields) == len(input_dict):
        raise HTTPException(
            status_code=400,
            detail="At least one feature must be provided"
        )


def _validate_request_covers_model_features(input_dict: dict, feature_names: list[str]) -> None:
    # Сheck: if model was retrained with new feature names
    # but PredictionRequest schema wasn't updated accordingly
    absent_features  = [
        feature for feature in feature_names
        if feature not in input_dict
    ]
    if absent_features :
        raise HTTPException(
            status_code=500,
            detail=f"Request schema does not match model features: {absent_features}"
        )


def _get_training_range_warnings(input_dict: dict, training_ranges: dict[str, dict[str, float]]) -> list[str]:
    warnings = []

    for feature, value in input_dict.items():
        if value is None:
            continue

        bounds = training_ranges.get(feature)
        if bounds is None:
            continue

        min_value = bounds.get("min")
        max_value = bounds.get("max")

        if min_value is not None and value < min_value:
            warnings.append(
                f"{feature}={value} is below training range [{min_value}, {max_value}]"
            )
        elif max_value is not None and value > max_value:
            warnings.append(
                f"{feature}={value} is above training range [{min_value}, {max_value}]"
            )

    return warnings


def _postprocess_prediction(raw_prediction: float) -> tuple[float, list[str]]:
    prediction = raw_prediction
    warnings: list[str] = []

    if prediction > PLAUSIBLE_MAX:
        prediction = PLAUSIBLE_MAX
        warnings.append(f"Raw model output exceeded the plausible upper bound ({PLAUSIBLE_MAX}) and was clipped.")

    if prediction < PLAUSIBLE_MIN:
        prediction = PLAUSIBLE_MIN
        warnings.append(f"Raw model output was below the plausible lower bound ({PLAUSIBLE_MIN}) and was clipped.")

    return prediction, warnings


def _prepare_single_record(request: PredictionRequest, artifact: ModelArtifact) -> tuple[dict, list[str], list[str]]:
    input_dict = request.model_dump()

    missing_fields = _get_missing_fields(input_dict)
    _validate_not_all_missing(input_dict, missing_fields)
    _validate_request_covers_model_features(input_dict, artifact.feature_names)
    warnings = _get_training_range_warnings(input_dict, artifact.training_ranges)

    return input_dict, missing_fields, warnings


def make_prediction(request: PredictionRequest, artifact: ModelArtifact) -> tuple[float, list[str], list[str]]:
    try:
        input_dict, missing_fields, warnings = _prepare_single_record(request, artifact)

        df = pd.DataFrame([input_dict])
        df = df[artifact.feature_names]

        raw_prediction = float(artifact.pipeline.predict(df)[0])
        prediction, prediction_warnings = _postprocess_prediction(raw_prediction)
        warnings.extend(prediction_warnings)

        return prediction, missing_fields, warnings

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        raise HTTPException(
            status_code=500, 
            detail="Unexpected error during prediction"
        )
    

def make_batch_prediction(requests: list[PredictionRequest], artifact: ModelArtifact) -> list[dict]:
    try:
        records: list[dict] = []
        missing_fields_list: list[list[str]] = []
        warnings_list: list[list[str]] = []

        for request in requests:
            input_dict, missing_fields, warnings = _prepare_single_record(request, artifact)
            records.append(input_dict)
            missing_fields_list.append(missing_fields)
            warnings_list.append(warnings)

        df = pd.DataFrame(records)
        df = df[artifact.feature_names]

        raw_predictions = artifact.pipeline.predict(df)
        
        results: list[dict] = []
        for raw_prediction, missing_fields, warnings in zip(
            raw_predictions,
            missing_fields_list,
            warnings_list,
            strict=True,
        ):
            prediction, prediction_warnings = _postprocess_prediction(
                float(raw_prediction)
            )

            results.append(
                {
                    "prediction": prediction,
                    "imputed_fields": missing_fields,
                    "warnings": warnings + prediction_warnings,
                }
            )

        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Batch prediction failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Unexpected error during batch prediction",
        )