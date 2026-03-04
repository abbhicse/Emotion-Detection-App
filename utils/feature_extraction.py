import mediapipe as mp
import cv2
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

def extract_landmarks(image):
    """
    Use MediaPipe to extract facial landmarks (optional).
    Works with both PIL.Image and NumPy arrays.
    Returns list of (x, y) landmark coordinates.
    """
    # ✅ Convert PIL image to NumPy if needed
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # ✅ Ensure image has 3 channels (MediaPipe requires RGB)
    if image.ndim == 2:  # grayscale → stack channels
        image = np.stack([image]*3, axis=-1)

    # ✅ Convert RGB (PIL) → BGR (OpenCV)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(image_bgr)

        if results.multi_face_landmarks:
            landmarks = [(lm.x, lm.y) for lm in results.multi_face_landmarks[0].landmark]
            return landmarks

    return []
