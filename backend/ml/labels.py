# Ordered exactly as in the PyTorch model's output layer
CLASSES = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# Mapping from class index to human-readable condition
def get_class_info(class_idx: int) -> dict:
    if class_idx < 0 or class_idx >= len(CLASSES):
        return {"crop": "Unknown", "condition": "Unknown", "is_healthy": False}
        
    class_name = CLASSES[class_idx]
    
    # Extract crop name
    crop = "Unknown"
    if class_name.startswith("Pepper"):
        crop = "Pepper"
    elif class_name.startswith("Potato"):
        crop = "Potato"
    elif class_name.startswith("Tomato"):
        crop = "Tomato"
        
    # Is healthy?
    is_healthy = "healthy" in class_name.lower()
    
    # Format condition name cleanly
    # Example: "Tomato_Spider_mites_Two_spotted_spider_mite" -> "Spider Mites Two Spotted Spider Mite"
    condition = class_name
    # Strip crop prefix and underscores
    for prefix in ["Pepper__bell___", "Potato___", "Tomato_", "Tomato__"]:
        if condition.startswith(prefix):
            condition = condition.replace(prefix, "", 1)
            break
            
    condition = condition.replace("_", " ").strip()
    
    # Capitalize each word nicely
    condition = " ".join([word.capitalize() for word in condition.split()])
    
    # Combine with crop name for final condition description
    full_condition = f"{crop} {condition}"
    
    return {
        "crop": crop,
        "condition": full_condition,
        "is_healthy": is_healthy
    }
