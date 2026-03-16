import cv2
import numpy as np
from PIL import Image


def _resize_face(face_array):
    if face_array is None or face_array.size == 0:
        return None
    try:
        face_resized = cv2.resize(face_array, (48, 48))
        return Image.fromarray(face_resized)
    except Exception as e:
        print(f"[Resize Error] {e}")
        return None


def _detect_with_mediapipe(img_rgb, model_selection=0, min_conf=0.3):
    from mediapipe.python.solutions import face_detection as mp_face_detection
    from mediapipe.python.solutions import face_mesh as mp_face_mesh

    h, w, _ = img_rgb.shape

    with mp_face_detection.FaceDetection(
        model_selection=model_selection,
        min_detection_confidence=min_conf
    ) as detector, mp_face_mesh.FaceMesh(
        static_image_mode=True
    ) as face_mesh:

        detection_results = detector.process(img_rgb)
        if not detection_results.detections:
            return None

        bbox = detection_results.detections[0].location_data.relative_bounding_box
        x1 = max(0, int(bbox.xmin * w))
        y1 = max(0, int(bbox.ymin * h))
        x2 = min(w, int((bbox.xmin + bbox.width) * w))
        y2 = min(h, int((bbox.ymin + bbox.height) * h))

        # add a little padding
        pad_x = int(0.08 * (x2 - x1))
        pad_y = int(0.08 * (y2 - y1))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        mesh_results = face_mesh.process(img_rgb)

        if mesh_results.multi_face_landmarks:
            landmarks = mesh_results.multi_face_landmarks[0].landmark
            left_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
            right_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])

            dx, dy = right_eye - left_eye
            angle = np.degrees(np.arctan2(dy, dx))
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            aligned = cv2.warpAffine(img_rgb, rot_mat, (w, h))
            face_crop = aligned[y1:y2, x1:x2]
        else:
            face_crop = img_rgb[y1:y2, x1:x2]

        return _resize_face(face_crop)


def _detect_with_opencv(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face_crop = img_rgb[y:y+h, x:x+w]
    return _resize_face(face_crop)


def detect_face(image):
    """
    Returns a 48x48 RGB PIL face image or None.
    """

    if image.mode != "RGB":
        image = image.convert("RGB")

    w, h = image.size

    # already likely a cropped face
    if min(w, h) <= 96:
        return image.resize((48, 48))

    # upscale a bit for weak detectors
    min_size = 200
    if w < min_size or h < min_size:
        image = image.resize((max(w, min_size), max(h, min_size)), resample=Image.BICUBIC)

    img_rgb = np.array(image)

    # Try MediaPipe short-range first
    try:
        face = _detect_with_mediapipe(img_rgb, model_selection=0, min_conf=0.3)
        if face is not None:
            return face
    except Exception as e:
        print(f"[MediaPipe short-range error] {e}")

    # Then MediaPipe long-range
    try:
        face = _detect_with_mediapipe(img_rgb, model_selection=1, min_conf=0.3)
        if face is not None:
            return face
    except Exception as e:
        print(f"[MediaPipe long-range error] {e}")

    # Finally OpenCV fallback
    try:
        face = _detect_with_opencv(img_rgb)
        if face is not None:
            return face
    except Exception as e:
        print(f"[OpenCV fallback error] {e}")

    return None
