import torch
import torch.nn.functional as F
from typing import Dict, Any

from .model import CropDiseaseClassifier
from .labels import get_class_info
from .preprocessing import preprocess_image, validate_image
from .explainability import GradCAM, create_gradcam_overlay

# Singleton instance to avoid reloading the model for every request
_classifier = None
_gradcam = None

def load_model(model_path: str):
    """
    Load the model once at startup.
    """
    global _classifier, _gradcam
    _classifier = CropDiseaseClassifier()
    _classifier.load(model_path)
    _gradcam = GradCAM(_classifier.model)
    print("MobileNetV2 model loaded successfully.")

def predict_image(img, confidence_threshold: float = 0.40) -> Dict[str, Any]:
    """
    Perform preprocessing and inference on a PIL Image object (e.g. pre-cropped leaf region).
    """
    if _classifier is None:
        raise RuntimeError("Model is not loaded. Call load_model() first.")
        
    tensor = preprocess_image(img)
    
    # Run prediction
    logits = _classifier.predict(tensor)
    probs = F.softmax(logits, dim=1)[0]
    
    # Get top 3 predictions
    top_probs, top_indices = torch.topk(probs, 3)
    
    top_predictions = []
    for i in range(3):
        prob = top_probs[i].item()
        idx = top_indices[i].item()
        info = get_class_info(idx)
        top_predictions.append({
            "condition": info["condition"],
            "score": prob
        })
        
    # Get main prediction
    main_prob = top_probs[0].item()
    main_idx = top_indices[0].item()
    main_info = get_class_info(main_idx)
    
    result = {
        "crop": main_info["crop"],
        "condition": main_info["condition"],
        "is_healthy": main_info["is_healthy"],
        "confidence": main_prob,
        "model": "MobileNetV2",
        "severity": None,
        "affected_region": None,
        "recommendation_available": True,
        "top_predictions": top_predictions,
        "below_threshold": main_prob < confidence_threshold
    }
    
    return result

def predict(image_bytes: bytes, confidence_threshold: float = 0.40) -> Dict[str, Any]:
    """
    Perform validation, preprocessing, and inference on the image bytes.
    Returns a structured dictionary with the predictions.
    """
    if _classifier is None:
        raise RuntimeError("Model is not loaded. Call load_model() first.")
        
    is_valid, error_msg, img = validate_image(image_bytes)
    if not is_valid:
        return {"error": error_msg}
        
    return predict_image(img, confidence_threshold=confidence_threshold)


def explain(image_bytes: bytes) -> str:
    """
    Generate a Grad-CAM overlay for the given image and return it as a base64 string.
    """
    if _classifier is None or _gradcam is None:
        raise RuntimeError("Model is not loaded.")
        
    is_valid, error_msg, img = validate_image(image_bytes)
    if not is_valid:
        raise ValueError(error_msg)
        
    tensor = preprocess_image(img)
    tensor = tensor.to(_classifier.device)
    
    cam = _gradcam.generate(tensor)
    return create_gradcam_overlay(image_bytes, cam)
