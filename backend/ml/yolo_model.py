import io
import logging
import os
import base64
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

_detector = None


def is_loaded() -> bool:
    return _detector is not None


def load_model(model_path: str) -> bool:
    """Load the YOLO detector once and keep it available for requests."""
    global _detector

    if not os.path.isfile(model_path):
        logger.error("YOLO model file was not found at %s", model_path)
        return False

    try:
        from ultralytics import YOLO

        _detector = YOLO(model_path)
        logger.info("YOLO detector loaded successfully from %s", model_path)
        return True
    except Exception:
        _detector = None
        logger.exception("Unable to load YOLO detector from %s", model_path)
        return False


def _to_model_input(image_path_or_array: Any) -> Any:
    if isinstance(image_path_or_array, (bytes, bytearray)):
        with Image.open(io.BytesIO(image_path_or_array)) as image:
            return np.asarray(image.convert("RGB"))
    return image_path_or_array


def detect(image_path_or_array: Any) -> List[Dict[str, Any]]:
    """Return pixel-coordinate ``xyxy`` boxes for one image."""
    if _detector is None:
        return []

    results = _detector.predict(source=_to_model_input(image_path_or_array), imgsz=640, verbose=False)
    if not results:
        return []

    result = results[0]
    names = result.names
    detections: List[Dict[str, Any]] = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        detections.append({
            "class_name": str(names[class_id]),
            "confidence": float(box.conf.item()),
            "bbox": [float(value) for value in box.xyxy[0].tolist()],
        })
    return detections


def create_annotation(image_bytes: bytes, detections: List[Dict[str, Any]]) -> str:
    """Draw detector boxes and labels and return a base64 PNG for the API."""
    with Image.open(io.BytesIO(image_bytes)) as image:
        annotated = image.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        label = f'{detection["class_name"]} {detection["confidence"]:.0%}'
        draw.rectangle((x1, y1, x2, y2), outline="#ef4444", width=max(3, annotated.width // 300))
        text_box = draw.textbbox((x1, y1), label)
        draw.rectangle(text_box, fill="#ef4444")
        draw.text((x1, y1), label, fill="white")

    output = io.BytesIO()
    annotated.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def estimate_severity(
    detections: List[Dict[str, Any]], image_width: int, image_height: int
) -> Dict[str, Any]:
    """Estimate severity from the sum of detection-box areas.

    This is a box-area approximation, not true pixel-level segmentation.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")

    image_area = image_width * image_height
    detected_area = 0.0
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        box_width = max(0.0, min(float(image_width), x2) - max(0.0, x1))
        box_height = max(0.0, min(float(image_height), y2) - max(0.0, y1))
        detected_area += box_width * box_height

    ratio = min(detected_area / image_area, 1.0)
    percentage = ratio * 100
    if not detections:
        level = "None"
    elif percentage < 5:
        level = "Mild"
    elif percentage <= 20:
        level = "Moderate"
    else:
        level = "Severe"

    return {"ratio": ratio, "level": level}