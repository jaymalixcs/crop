import os
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from typing import List

from ..database import db, models
from ..schemas.analysis import AnalysisResultSchema, DashboardStatsSchema, RecommendationSchema
from ..services.image_service import save_upload_file
from ..services.analysis_service import process_analysis, get_recommendations
from ml.explainability import create_gradcam_overlay
from ml.inference import explain
from ml.yolo_model import is_loaded as yolo_is_loaded

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.get("/model/info")
def model_info():
    return {
        "model": "MobileNetV2",
        "type": "Image Classification",
        "classes": 15,
        "input_size": "224x224",
        "version": "1.0",
        "detector": {
            "model": "YOLOv8s",
            "type": "Object Detection",
            "loaded": yolo_is_loaded(),
            "input_size": "640x640",
            "classes": 8
        }
    }

@router.post("/analyze", response_model=AnalysisResultSchema)
async def analyze_image(file: UploadFile = File(...), session: Session = Depends(db.get_db)):
    """Upload an image and get crop health analysis."""
    # 1. Save upload file
    image_bytes, filename = await save_upload_file(file)
    
    # 2. Process analysis
    result = await process_analysis(image_bytes, filename, session)
    return result

@router.get("/analysis/{analysis_id}", response_model=AnalysisResultSchema)
def get_analysis(analysis_id: int, session: Session = Depends(db.get_db)):
    """Get details of a past analysis."""
    record = session.query(models.AnalysisRecord).filter(models.AnalysisRecord.id == analysis_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")
        
    recs = get_recommendations(record.condition)
    
    return AnalysisResultSchema(
        id=record.id,
        crop=record.crop,
        condition=record.condition,
        is_healthy="healthy" in record.condition.lower(),
        confidence=record.confidence,
        model_score=record.confidence,
        top_predictions=record.top_predictions,
        recommendations=RecommendationSchema(**recs),
        explainability_available=True,
        image_url=f"/uploads/{record.image_path}",
        created_at=record.created_at
    )

@router.get("/history", response_model=List[AnalysisResultSchema])
def get_history(skip: int = 0, limit: int = 50, session: Session = Depends(db.get_db)):
    """Get history of analyses."""
    records = session.query(models.AnalysisRecord).order_by(models.AnalysisRecord.created_at.desc()).offset(skip).limit(limit).all()
    
    results = []
    for record in records:
        recs = get_recommendations(record.condition)
        results.append(
            AnalysisResultSchema(
                id=record.id,
                crop=record.crop,
                condition=record.condition,
                is_healthy="healthy" in record.condition.lower(),
                confidence=record.confidence,
                model_score=record.confidence,
                top_predictions=record.top_predictions,
                recommendations=RecommendationSchema(**recs),
                explainability_available=True,
                image_url=f"/uploads/{record.image_path}",
                created_at=record.created_at
            )
        )
    return results

@router.get("/dashboard", response_model=DashboardStatsSchema)
def get_dashboard(session: Session = Depends(db.get_db)):
    """Get statistics for the dashboard."""
    total = session.query(models.AnalysisRecord).count()
    if total == 0:
        return DashboardStatsSchema(total_analyses=0, healthy_count=0, attention_needed_count=0)
        
    healthy_count = 0
    attention_count = 0
    
    # Simple counting (can be optimized with SQL aggregations)
    records = session.query(models.AnalysisRecord).all()
    
    crop_counts = {}
    condition_counts = {}
    
    for r in records:
        if "healthy" in r.condition.lower():
            healthy_count += 1
        else:
            attention_count += 1
            
        crop_counts[r.crop] = crop_counts.get(r.crop, 0) + 1
        condition_counts[r.condition] = condition_counts.get(r.condition, 0) + 1
        
    most_detected_crop = max(crop_counts, key=crop_counts.get) if crop_counts else None
    
    # Get most detected condition (excluding healthy)
    sick_conditions = {k: v for k, v in condition_counts.items() if "healthy" not in k.lower()}
    most_detected_condition = max(sick_conditions, key=sick_conditions.get) if sick_conditions else None
    
    return DashboardStatsSchema(
        total_analyses=total,
        healthy_count=healthy_count,
        attention_needed_count=attention_count,
        most_detected_crop=most_detected_crop,
        most_detected_condition=most_detected_condition
    )

@router.get("/explain/{analysis_id}")
def get_explanation(analysis_id: int, session: Session = Depends(db.get_db)):
    """Run Grad-CAM on a past image."""
    record = session.query(models.AnalysisRecord).filter(models.AnalysisRecord.id == analysis_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")
        
    from ..core.config import settings
    file_path = os.path.join(settings.UPLOAD_DIR, record.image_path)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image file not found")
        
    with open(file_path, "rb") as f:
        image_bytes = f.read()
        
    try:
        b64_img = explain(image_bytes)
        return {"overlay_base64": b64_img}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@router.get("/crops")
def get_crops():
    """Return supported crops and conditions."""
    return {
        "Tomato": [
            "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Late Blight",
            "Tomato Leaf Mold", "Tomato Septoria Leaf Spot", "Tomato Spider Mites Two Spotted Spider Mite",
            "Tomato Target Spot", "Tomato Tomato Yellowleaf Curl Virus", "Tomato Tomato Mosaic Virus",
            "Tomato Healthy"
        ],
        "Potato": ["Potato Early Blight", "Potato Late Blight", "Potato Healthy"],
        "Pepper": ["Pepper Bacterial Spot", "Pepper Healthy"]
    }

@router.get("/evaluation")
def get_evaluation(force_rerun: bool = False):
    """Return comprehensive evaluation metrics across all 17 benchmark categories."""
    from ..services.evaluation_service import get_or_run_evaluations
    return get_or_run_evaluations(force_rerun=force_rerun)

@router.post("/evaluation/run")
def run_evaluation():
    """Trigger a fresh execution of the complete evaluation suite."""
    from ..services.evaluation_service import get_or_run_evaluations
    return get_or_run_evaluations(force_rerun=True)

