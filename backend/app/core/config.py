import os
from pydantic import BaseModel
from typing import List

class Settings(BaseModel):
    # Application Config
    PROJECT_NAME: str = "Crop Health AI"
    
    # Model Config
    MODEL_PATH: str = os.getenv("MODEL_PATH", "../models/multicrop_classifier.pt")
    YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "../models/yolo_detector.pt")
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.40"))
    
    # Storage Config
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
    MAX_UPLOAD_SIZE: int = int(os.getenv("MAX_UPLOAD_SIZE", str(10 * 1024 * 1024))) # 10MB
    
    # Database Config
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./crop_health.db")
    
    # CORS
    CORS_ORIGINS: List[str] = ["*"] # Allow all for demo purposes

settings = Settings()
