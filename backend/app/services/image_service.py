import os
import uuid
from fastapi import UploadFile, HTTPException
from ..core.config import settings

# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

async def save_upload_file(upload_file: UploadFile) -> str:
    """
    Saves an uploaded file to the configured uploads directory with a safe UUID filename.
    Returns the relative path to the saved file.
    """
    if not upload_file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
        
    # Get extension securely
    _, ext = os.path.splitext(upload_file.filename)
    if not ext:
        ext = ".jpg" # fallback
        
    ext = ext.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file extension: {ext}")
        
    # Generate unique safe filename
    filename = f"{uuid.uuid4()}{ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    
    # Check size if possible before reading all, or just read and check
    content = await upload_file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large. Maximum size is {settings.MAX_UPLOAD_SIZE / (1024*1024)} MB.")
        
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="File is empty.")
        
    # Save file
    with open(file_path, 'wb') as out_file:
        out_file.write(content)
        
    # Return the bytes for immediate processing, and the path for DB
    return content, filename
