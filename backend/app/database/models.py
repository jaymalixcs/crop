from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from datetime import datetime
from .db import Base
import json

class AnalysisRecord(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    image_path = Column(String, index=True)
    crop = Column(String, index=True)
    condition = Column(String, index=True)
    confidence = Column(Float)
    model_name = Column(String)
    notes = Column(Text, nullable=True)
    
    # Store top predictions as a JSON string
    _top_predictions_json = Column("top_predictions", Text)

    @property
    def top_predictions(self):
        if self._top_predictions_json:
            return json.loads(self._top_predictions_json)
        return []

    @top_predictions.setter
    def top_predictions(self, value):
        self._top_predictions_json = json.dumps(value)
