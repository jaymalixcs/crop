import torch
import torch.nn as nn
from torchvision import models
from .base import VisionModel

class CropDiseaseClassifier(VisionModel):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
    def load(self, model_path: str) -> None:
        """
        Loads the MobileNetV2 architecture, modifies the classifier head for 15 classes,
        and loads the saved state dictionary.
        """
        # Load base MobileNetV2 without pretrained weights
        self.model = models.mobilenet_v2()
        
        # Modify the classifier head for 15 classes (as done in training)
        num_classes = 15
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

    def predict(self, tensor: torch.Tensor):
        """
        Runs the tensor through the model and returns raw logits.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load() first.")
            
        tensor = tensor.to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
        return logits
