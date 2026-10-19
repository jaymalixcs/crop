"""
test_suite.py
~~~~~~~~~~~~~
Automated Pytest unit and integration test suite.
Run with: pytest backend/tests/test_suite.py
"""

import io
import os
import sys
import pytest
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from app.main import app
from ml.preprocessing import validate_image, preprocess_image
from ml.labels import CLASSES, get_class_info
from ml.yolo_model import estimate_severity
from app.services.analysis_service import get_recommendations
from app.services.evaluation_service import wilson_score_interval

client = TestClient(app)

# ---------------------------------------------------------------------------
# Unit Tests: Preprocessing & Validation
# ---------------------------------------------------------------------------

def test_validate_image_valid_rgb():
    img = Image.new("RGB", (224, 224), color=(34, 139, 34))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 150, 150], fill=(120, 200, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    is_valid, msg, pil_img = validate_image(buf.getvalue())
    assert is_valid is True
    assert msg is None
    assert pil_img is not None


def test_validate_image_empty():
    is_valid, msg, _ = validate_image(b"")
    assert is_valid is False
    assert "empty" in msg.lower()

def test_validate_image_corrupted():
    is_valid, msg, _ = validate_image(b"not a valid image byte stream")
    assert is_valid is False
    assert "valid image" in msg.lower()

def test_validate_image_blank_solid():
    img = Image.new("RGB", (224, 224), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    is_valid, msg, _ = validate_image(buf.getvalue())
    assert is_valid is False
    assert "blank" in msg.lower()

# ---------------------------------------------------------------------------
# Unit Tests: Labels & Taxonomy
# ---------------------------------------------------------------------------

def test_labels_classes_count():
    assert len(CLASSES) == 15

def test_get_class_info_formatting():
    info = get_class_info(0)
    assert info["crop"] == "Pepper"
    assert "Bacterial" in info["condition"]

    tomato_info = get_class_info(6)
    assert tomato_info["crop"] == "Tomato"
    assert "Early Blight" in tomato_info["condition"]

# ---------------------------------------------------------------------------
# Unit Tests: Severity Estimation
# ---------------------------------------------------------------------------

def test_severity_empty():
    res = estimate_severity([], 640, 640)
    assert res["level"] == "None"
    assert res["ratio"] == 0.0

def test_severity_tiers():
    mild = estimate_severity([{"bbox": [10, 10, 50, 50]}], 640, 640)
    assert mild["level"] == "Mild"

    severe = estimate_severity([{"bbox": [0, 0, 400, 400]}], 640, 640)
    assert severe["level"] == "Severe"

# ---------------------------------------------------------------------------
# Unit Tests: Recommendations Integrity
# ---------------------------------------------------------------------------

def test_recommendations_known():
    rec = get_recommendations("Tomato Early Blight")
    assert rec["crop"] == "Tomato"
    assert len(rec["general_management"]) > 0
    assert len(rec["prevention"]) > 0
    assert len(rec["monitoring"]) > 0

def test_recommendations_fallback():
    rec = get_recommendations("Unknown Disease Condition XYZ")
    assert rec["crop"] == "Unknown"
    assert rec["expert_verification"] is True

# ---------------------------------------------------------------------------
# Unit Tests: Wilson Score Confidence Interval
# ---------------------------------------------------------------------------

def test_wilson_interval():
    ci = wilson_score_interval(954, 1000, 0.95)
    assert ci["proportion"] == 0.954
    assert ci["lower"] > 0.93
    assert ci["upper"] < 0.97
    assert ci["error_margin"] > 0

# ---------------------------------------------------------------------------
# API Tests
# ---------------------------------------------------------------------------

def test_api_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}

def test_api_model_info():
    res = client.get("/api/model/info")
    assert res.status_code == 200
    data = res.json()
    assert data["model"] == "MobileNetV2"
    assert data["classes"] == 15
    assert data["detector"]["model"] == "YOLOv8s"

def test_api_crops():
    res = client.get("/api/crops")
    assert res.status_code == 200
    crops = res.json()
    assert "Tomato" in crops
    assert "Potato" in crops
    assert "Pepper" in crops

def test_api_evaluation():
    res = client.get("/api/evaluation")
    assert res.status_code == 200
    eval_data = res.json()
    assert eval_data["metadata"]["status"] == "ALL_TESTS_PASSING"
    assert eval_data["summary"]["overall_accuracy"] >= 0.95
    assert "must_have" in eval_data
    assert "should_have" in eval_data
    assert "software_testing" in eval_data
    assert "optional" in eval_data
