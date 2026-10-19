import io
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as transforms
import torch
from typing import Tuple, Optional

# Standard MobileNetV2 preprocessing
IMG_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def validate_image(image_bytes: bytes) -> Tuple[bool, Optional[str], Optional[Image.Image]]:
    """
    Check if the uploaded image is valid and of reasonable quality.
    Returns (is_valid, error_message, PIL_Image)
    """
    if not image_bytes:
        return False, "Image file is empty.", None
        
    try:
        # Load image
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # verify format is valid
        
        # Re-open after verify to perform operations
        img = Image.open(io.BytesIO(image_bytes))
        
        # Check format
        if img.format not in ['JPEG', 'PNG', 'WEBP', 'JPG']:
            return False, f"Unsupported image format: {img.format}. Please upload JPG, PNG, or WEBP.", None
            
        # Check resolution (too small)
        width, height = img.size
        if width < 50 or height < 50:
            return False, "Image resolution is too low. Please upload a clearer photo.", None
            
        # Optional: check if image is completely black or white using extrama
        extrema = img.convert("L").getextrema()
        if extrema[0] == extrema[1]:
            return False, "Image appears to be completely blank or corrupted.", None

        # Convert to RGB (in case of RGBA or Grayscale)
        img = img.convert('RGB')
        
        return True, None, img
        
    except UnidentifiedImageError:
        return False, "The uploaded file is not a valid image.", None
    except Exception as e:
        return False, f"Could not process image: {str(e)}", None

def preprocess_image(img: Image.Image) -> torch.Tensor:
    """
    Convert a PIL image to a normalized PyTorch tensor ready for MobileNetV2.
    Adds the batch dimension (1, C, H, W).
    """
    tensor = transform(img)
    return tensor.unsqueeze(0)
