# utils/image_utils.py

import numpy as np
from PIL import Image

def validate_image(uploaded_file, max_size=5*1024*1024):
    """
    Validate uploaded image:
    - Must be JPEG/PNG
    - Must be < 5 MB
    """
    if uploaded_file.size > max_size:
        return False
    if uploaded_file.type not in ["image/jpeg", "image/png"]:
        return False
    return True


def load_image(uploaded_file):
    """
    Load image from uploaded file as a PIL Image (converted to RGB).
    This ensures compatibility with face detection and transforms.
    """
    try:
        image = Image.open(uploaded_file)
        image.load()  # Force actual image loading
        #print(f"[DEBUG] Loaded image mode: {image.mode}")
        return image  # ✅ Return PIL Image, not NumPy
    except Exception as e:
        print(f"[Image Load Error] {e}")
        return None
