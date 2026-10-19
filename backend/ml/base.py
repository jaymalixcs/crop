from abc import ABC, abstractmethod
from typing import Any, Dict

class VisionModel(ABC):
    """
    Base interface for all computer vision models in the system.
    This abstraction allows future models (e.g., Object Detection, Segmentation)
    to be plugged in seamlessly.
    """
    
    @abstractmethod
    def load(self, model_path: str) -> None:
        """Load model weights and initialize architecture."""
        pass

    @abstractmethod
    def predict(self, image: Any) -> Dict[str, Any]:
        """
        Run inference on the provided image and return a structured result.
        """
        pass
