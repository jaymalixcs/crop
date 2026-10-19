import io
import base64
from typing import Dict, Any
from PIL import Image
from ultralytics import YOLO

_detector = None

# Must match the class order used during training
CLASS_NAMES = [
    "Tomato_Early_Blight", "Tomato_Healthy", "Tomato_Late_Blight", "Tomato_Septoria_Leaf_Spot",
    "Potato_Early_Blight", "Potato_Late_Blight",
    "Pepper_Bacterial", "Pepper_Healthy",
]

def load_yolo_model(model_path: str):
    """Load the YOLOv8 detector once at startup."""
    global _detector
    _detector = YOLO(model_path)
    print("YOLOv8 detector loaded successfully.")

def _estimate_severity(boxes, img_w: int, img_h: int) -> Dict[str, Any]:
    """Box-area approximation of affected leaf area (not true pixel segmentation)."""
    image_area = img_w * img_h
    disease_area = 0.0
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        disease_area += (x2 - x1) * (y2 - y1)

    ratio = disease_area / image_area if image_area else 0.0
    if ratio < 0.05:
        level = "Mild"
    elif ratio < 0.20:
        level = "Moderate"
    else:
        level = "Severe"

    return {"ratio": ratio, "level": level}

def detect(image_bytes: bytes, confidence_threshold: float = 0.25) -> Dict[str, Any]:
    """
    Run YOLO detection on an image and return structured detections,
    severity, and a base64-encoded image with boxes drawn on it.
    """
    if _detector is None:
        raise RuntimeError("YOLO model is not loaded. Call load_yolo_model() first.")

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = _detector.predict(img, conf=confidence_threshold, verbose=False)
    result = results[0]

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        detections.append({
            "class_name": CLASS_NAMES[cls_id],
            "confidence": float(box.conf[0]),
            "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],  # [x1, y1, x2, y2]
        })

    severity = _estimate_severity(result.boxes, img.width, img.height)

    # Draw boxes on the image (BGR numpy array -> RGB -> PNG -> base64)
    annotated = result.plot()[..., ::-1]  # BGR to RGB
    annotated_img = Image.fromarray(annotated)
    buf = io.BytesIO()
    annotated_img.save(buf, format="PNG")
    annotated_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "detections": detections,
        "severity": severity,
        "annotated_image_base64": annotated_b64,
    }