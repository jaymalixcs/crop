from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class PredictionSchema(BaseModel):
    condition: str
    score: float

class RecommendationSchema(BaseModel):
    crop: str
    condition: str
    description: str
    general_management: List[str]
    prevention: List[str]
    monitoring: List[str]
    expert_verification: bool

class DetectionBox(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]

class SeverityResult(BaseModel):
    ratio: float
    level: str

class DetectorPrediction(BaseModel):
    class_name: str
    confidence: float

class AnalysisResultSchema(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    id: Optional[int] = None
    crop: str
    condition: str
    is_healthy: bool
    confidence: float
    model_score: float  # Alias for confidence in the frontend
    top_predictions: List[PredictionSchema]
    detections: List[DetectionBox] = Field(default_factory=list)
    severity: Optional[SeverityResult] = None
    detection_severity: Optional[SeverityResult] = None
    detector_prediction: Optional[DetectorPrediction] = None
    annotated_image_base64: Optional[str] = None
    classifier_condition: Optional[str] = None
    classifier_crop: Optional[str] = None
    classifier_confidence: Optional[float] = None
    recommendations: Optional[RecommendationSchema] = None
    explainability_available: bool = True
    below_threshold: bool = False
    image_url: Optional[str] = None
    created_at: Optional[datetime] = None
    
class DashboardStatsSchema(BaseModel):
    total_analyses: int
    healthy_count: int
    attention_needed_count: int
    most_detected_crop: Optional[str] = None
    most_detected_condition: Optional[str] = None
