import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import base64
import io

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.feature_extractor = model.features
        self.gradients = None
        self.activations = None
        
        # Register hooks into the last layer of features in MobileNetV2
        self.feature_extractor[-1].register_forward_hook(self.save_activation)
        self.feature_extractor[-1].register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        
        # Enable gradient tracking
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        # Get gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Calculate weights (global average pooling of gradients)
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weight the activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        cam = cam - np.min(cam)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
            
        # Resize to input size
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        
        return cam

def create_gradcam_overlay(image_bytes: bytes, cam: np.ndarray) -> str:
    """
    Creates a heatmap overlay on the original image and returns it as a base64 JPEG.
    """
    # Open original image and resize to match model input (224x224)
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_np = np.array(img)
    
    # Apply colormap to CAM
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    # Convert BGR to RGB
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    heatmap = np.float32(heatmap) / 255
    
    # Overlay heatmap on image
    cam_img = heatmap * 0.4 + np.float32(img_np) / 255 * 0.6
    cam_img = cam_img / np.max(cam_img)
    
    # Convert to PIL Image
    cam_img_uint8 = np.uint8(255 * cam_img)
    result_img = Image.fromarray(cam_img_uint8)
    
    # Save to base64
    buffered = io.BytesIO()
    result_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
