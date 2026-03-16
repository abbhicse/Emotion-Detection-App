import cv2
import numpy as np


def extract_landmarks(image):
    """
    Extract facial landmarks using MediaPipe Face Mesh.
    Accepts PIL.Image or NumPy array.
    Returns a list of (x, y) landmark coordinates.
    """
    try:
        from mediapipe.python.solutions import face_mesh as mp_face_mesh
    except Exception as e:
        print(f"[MediaPipe Import Error - feature_extraction] {e}")
        return []

    if not isinstance(image, np.ndarray):
        image = np.array(image)

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    # Ensure RGB for MediaPipe
    if image.shape[-1] == 3:
        image_rgb = image
    else:
        return []

    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            return [(lm.x, lm.y) for lm in results.multi_face_landmarks[0].landmark]

    return []
