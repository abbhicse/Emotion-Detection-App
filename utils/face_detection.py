import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import dlib

# Initialize detectors
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
dlib_detector = dlib.get_frontal_face_detector()

def detect_face(image):
    """
    Detect and align a face using either MediaPipe (RGB) or Dlib (grayscale).
    Skips detection entirely for small, pre-cropped face images (e.g., FER-2013).
    Returns a 48x48 RGB face image (PIL format) or None if no face is found.
    """

    w, h = image.size
    is_grayscale = image.mode == "L"

    # Case 0: Already cropped face (FER-like, e.g., 48x48 or 64x64)
    if min(w, h) <= 64:
        #print("[DEBUG] Small image detected (likely already a face). Skipping detection.")
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image.resize((48, 48))

    # Case 1: Grayscale full photo → use Dlib
    if is_grayscale:
        #print("[DEBUG] Using Dlib for grayscale image")

        # Convert to NumPy grayscale array
        img_gray = np.array(image)

        # Detect faces
        faces = dlib_detector(img_gray)
        #print(f"[DEBUG] Dlib detected {len(faces)} face(s)")
        if not faces:
            #print("[DEBUG] Dlib: No face detected.")
            return None

        # Use the first face
        face = faces[0]
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        h, w = img_gray.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Crop and resize
        face_crop = img_gray[y1:y2, x1:x2]
        try:
            face_resized = cv2.resize(face_crop, (48, 48))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(face_rgb)
        except Exception as e:
            print(f"[Dlib Error] Resize failed: {e}")
            return None

    # Case 2: RGB image → use MediaPipe
    #print("[DEBUG] Using MediaPipe for RGB image")

    if image.mode != "RGB":
        image = image.convert("RGB")

    # Enlarge small images
    min_size = 150
    if image.size[0] < min_size or image.size[1] < min_size:
        #print(f"[DEBUG] Upscaling image from {image.size} to at least {min_size}px")
        new_size = (max(min_size, image.size[0]), max(min_size, image.size[1]))
        image = image.resize(new_size, resample=Image.BICUBIC)

    # Convert to NumPy RGB → BGR
    img_np = np.array(image)
    if img_np.ndim == 2:
        img_np = np.stack([img_np]*3, axis=-1)

    #print(f"[DEBUG] Image shape before MediaPipe: {img_np.shape}, dtype: {img_np.dtype}")
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h, w, _ = img_bgr.shape

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector, \
         mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:

        # Detect face
        detection_results = detector.process(img_bgr)
        if not detection_results.detections:
            #print("[DEBUG] MediaPipe: No face detected.")
            return None

        bbox = detection_results.detections[0].location_data.relative_bounding_box
        x1 = max(0, int(bbox.xmin * w))
        y1 = max(0, int(bbox.ymin * h))
        x2 = min(w, int((bbox.xmin + bbox.width) * w))
        y2 = min(h, int((bbox.ymin + bbox.height) * h))

        # Align face (optional)
        mesh_results = face_mesh.process(img_bgr)
        if mesh_results.multi_face_landmarks:
            landmarks = mesh_results.multi_face_landmarks[0].landmark
            left_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
            right_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
            dx, dy = right_eye - left_eye
            angle = np.degrees(np.arctan2(dy, dx))
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            img_bgr = cv2.warpAffine(img_bgr, rot_mat, (w, h))
            face_crop = img_bgr[y1:y2, x1:x2]
        else:
            face_crop = img_bgr[y1:y2, x1:x2]

        # Resize and convert
        try:
            face_resized = cv2.resize(face_crop, (48, 48))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            return Image.fromarray(face_rgb)
        except Exception as e:
            print(f"[MediaPipe Error] Resize/Crop failed: {e}")
            return None
