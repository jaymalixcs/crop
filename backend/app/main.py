from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from .core.config import settings
from .api.routes import router
from .database.db import engine, Base
from ml.inference import load_model as load_classifier_model
from ml.yolo_model import load_model as load_yolo_model

# Create DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AI-assisted crop disease and pest analysis platform",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for uploaded images
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

# Include routers
app.include_router(router, prefix="/api")

@app.on_event("startup")
def startup_event():
    """
    Load the ML model once when the application starts.
    """
    try:
        print(f"Loading model from {settings.MODEL_PATH}...")
        load_classifier_model(settings.MODEL_PATH)
    except Exception as e:
        print(f"Error loading model during startup: {e}")

    load_yolo_model(settings.YOLO_MODEL_PATH)
    print("Startup complete.")
