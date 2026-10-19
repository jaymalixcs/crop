"""
analysis_service.py
~~~~~~~~~~~~~~~~~~~
Orchestrates the full crop-disease analysis pipeline:

  1. Lesion-spot detection (OpenCV HSV/LAB segmentation).
  2. YOLOv8 leaf/disease bounding-box detection.
  3. MobileNetV2 classification on YOLO-cropped leaf ROI.
  4. Decision fusion: YOLO wins when confident, MobileNet wins for species.
  5. Annotated image with YOLO green box + red lesion circles.
  6. Recommendations fetch + SQLite persistence.
"""

import json
import logging
import os
from io import BytesIO
from PIL import Image
from sqlalchemy.orm import Session
from fastapi import HTTPException

from ml.inference import predict, predict_image
from ml.yolo_model import create_annotation, detect, estimate_severity
from ..database import models
from ..schemas.analysis import AnalysisResultSchema, RecommendationSchema
from ..core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Recommendations knowledge-base
# ---------------------------------------------------------------------------
RECOMMENDATIONS_FILE = os.path.join(
    os.path.dirname(__file__), '..', '..', 'knowledge', 'recommendations.json'
)
try:
    with open(RECOMMENDATIONS_FILE, 'r') as f:
        _recommendations_db = json.load(f)
except FileNotFoundError:
    _recommendations_db = []


def get_recommendations(condition: str) -> dict:
    """Return the recommendation record for *condition*, or a generic fallback."""
    for rec in _recommendations_db:
        if rec["condition"] == condition:
            return rec
    return {
        "crop": "Unknown",
        "condition": condition,
        "description": "Information for this specific condition is not available in our knowledge base.",
        "general_management": ["Consult local agricultural extension services for guidance."],
        "prevention": ["Maintain good field sanitation."],
        "monitoring": ["Regularly inspect plants for changes."],
        "expert_verification": True,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def process_analysis(image_bytes: bytes, filename: str, db: Session) -> AnalysisResultSchema:
    """
    Full analysis pipeline.

    Decision fusion rules (in priority order):
      A. If YOLO detects a leaf with confidence ≥ 0.70 AND MobileNet agrees on
         the crop species → trust YOLO's disease label, boost confidence.
      B. If YOLO detects a different crop species from MobileNet → trust
         MobileNet for the crop name but adopt YOLO's disease suffix.
      C. If YOLO confidence < 0.50 → rely solely on MobileNet.
    """
    from ml.lesion_detector import detect_disease_spots, draw_spot_circles

    detections: list = []
    detector_severity = None
    detector_prediction = None
    annotated_image_base64: str | None = None
    cropped_leaf_img = None
    yolo_bbox: list | None = None

    # ------------------------------------------------------------------
    # Step 1 – Open image
    # ------------------------------------------------------------------
    try:
        with Image.open(BytesIO(image_bytes)) as raw_image:
            image_rgb = raw_image.convert("RGB")
            image_width, image_height = image_rgb.size
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid or unreadable image: {exc}") from exc

    # ------------------------------------------------------------------
    # Step 2 – YOLOv8 detection
    # ------------------------------------------------------------------
    try:
        detections = detect(image_bytes)

        if detections:
            best = max(detections, key=lambda d: d["confidence"])
            detector_prediction = {
                "class_name": best["class_name"],
                "confidence": best["confidence"],
            }
            logger.info("YOLO best detection: %s @ %.1f%%",
                        best["class_name"], best["confidence"] * 100)

            bbox = best.get("bbox")
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                pad_w = max(5, int((x2 - x1) * 0.05))
                pad_h = max(5, int((y2 - y1) * 0.05))
                x1 = max(0, x1 - pad_w)
                y1 = max(0, y1 - pad_h)
                x2 = min(image_width,  x2 + pad_w)
                y2 = min(image_height, y2 + pad_h)
                yolo_bbox = [x1, y1, x2, y2]

                if (x2 - x1) > 20 and (y2 - y1) > 20:
                    cropped_leaf_img = image_rgb.crop((x1, y1, x2, y2))

        detector_severity = estimate_severity(detections, image_width, image_height)

    except Exception:
        logger.exception("YOLO detection failed; proceeding with MobileNet only")

    # ------------------------------------------------------------------
    # Step 3 – Lesion spot detection (restricted to YOLO ROI if available)
    # ------------------------------------------------------------------
    try:
        spots, spot_severity_ratio, _ = detect_disease_spots(image_bytes, yolo_bbox=yolo_bbox)
        logger.info("Lesion detector: %d spots, severity %.1f%%",
                    len(spots), spot_severity_ratio * 100)

        # Override YOLO severity if lesion-level analysis is more granular
        if spot_severity_ratio > 0:
            if spot_severity_ratio < 0.05:
                spot_level = "Mild"
            elif spot_severity_ratio <= 0.20:
                spot_level = "Moderate"
            else:
                spot_level = "Severe"
            detector_severity = {"ratio": spot_severity_ratio, "level": spot_level}
    except Exception:
        logger.exception("Lesion detector failed; continuing without spots")
        spots = []

    # ------------------------------------------------------------------
    # Step 4 – MobileNetV2 classification
    # ------------------------------------------------------------------
    if cropped_leaf_img is not None:
        logger.info("Classifying YOLO-cropped ROI (%dx%d)", *cropped_leaf_img.size)
        prediction_result = predict_image(cropped_leaf_img,
                                          confidence_threshold=settings.CONFIDENCE_THRESHOLD)
    else:
        logger.info("No YOLO crop available; classifying full image")
        prediction_result = predict(image_bytes, confidence_threshold=settings.CONFIDENCE_THRESHOLD)

    if "error" in prediction_result:
        raise HTTPException(status_code=400, detail=prediction_result["error"])

    # ------------------------------------------------------------------
    # Step 5 – Decision fusion
    # ------------------------------------------------------------------
    result_crop      = prediction_result["crop"]
    result_condition = prediction_result["condition"]
    result_is_healthy = prediction_result["is_healthy"]
    result_confidence = prediction_result["confidence"]

    if detector_prediction:
        yolo_conf   = detector_prediction["confidence"]
        yolo_label  = detector_prediction["class_name"].replace("_", " ")
        # First word of YOLO label is usually the crop species
        yolo_crop   = yolo_label.split(" ", 1)[0]

        if yolo_conf >= 0.50:
            if yolo_crop.lower() == result_crop.lower():
                # Same species: YOLO wins on condition, take max confidence
                result_condition  = yolo_label
                result_confidence = max(result_confidence, yolo_conf)
                logger.info("Fusion: YOLO wins (same species) → %s", result_condition)
            elif yolo_conf >= 0.70:
                # Different species but YOLO is very confident: keep MobileNet crop,
                # steal the disease suffix from YOLO
                disease_suffix = " ".join(yolo_label.split(" ")[1:]).strip()
                if disease_suffix:
                    result_condition = f"{result_crop} {disease_suffix}".strip()
                result_confidence = max(result_confidence, yolo_conf)
                logger.info("Fusion: species mismatch, adopting disease suffix → %s", result_condition)
            # else: YOLO conf < 0.50 or < 0.70 with mismatch → keep MobileNet

    # ------------------------------------------------------------------
    # Step 6 – Annotated image with YOLO box + lesion circles
    # ------------------------------------------------------------------
    try:
        annotated_image_base64 = draw_spot_circles(
            image_bytes,
            spots,
            result_condition,
            result_confidence,
            yolo_detections=detections,
        )
    except Exception:
        logger.exception("Annotation rendering failed")

    # ------------------------------------------------------------------
    # Step 7 – Recommendations & persistence
    # ------------------------------------------------------------------
    recs = get_recommendations(result_condition)

    db_record = models.AnalysisRecord(
        image_path=filename,
        crop=result_crop,
        condition=result_condition,
        confidence=result_confidence,
        model_name=prediction_result["model"],
        top_predictions=[p for p in prediction_result["top_predictions"]],
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)

    # ------------------------------------------------------------------
    # Step 8 – Build response
    # ------------------------------------------------------------------
    return AnalysisResultSchema(
        id=db_record.id,
        crop=result_crop,
        condition=result_condition,
        is_healthy=result_is_healthy,
        confidence=result_confidence,
        model_score=prediction_result["confidence"],
        top_predictions=prediction_result["top_predictions"],
        detections=detections,
        severity=detector_severity,
        detection_severity=detector_severity,
        detector_prediction=detector_prediction,
        annotated_image_base64=annotated_image_base64,
        classifier_condition=prediction_result["condition"],
        classifier_crop=prediction_result["crop"],
        classifier_confidence=prediction_result["confidence"],
        recommendations=RecommendationSchema(**recs),
        explainability_available=prediction_result["recommendation_available"],
        below_threshold=prediction_result["below_threshold"],
        image_url=f"/uploads/{filename}",
        created_at=db_record.created_at,
    )
