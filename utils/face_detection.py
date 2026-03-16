import cv2
import numpy as np
from PIL import Image


def detect_face(image):
    try:
        from mediapipe.python.solutions import face_detection as mp_face_detection
        from mediapipe.python.solutions import face_mesh as mp_face_mesh
    except Exception as e:
        print(f"[MediaPipe Import Error] {e}")
        return None

    w, h = image.size

    if min(w, h) <= 64:
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image.resize((48, 48))

    if image.mode != "RGB":
        image = image.convert("RGB")

    min_size = 150
    if image.size[0] < min_size or image.size[1] < min_size:
        new_size = (max(min_size, image.size[0]), max(min_size, image.size[1]))
        image = image.resize(new_size, resample=Image.BICUBIC)

    img_rgb = np.array(image)
    h, w, _ = img_rgb.shape

    with mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
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

        try:
            face_resized = cv2.resize(face_crop, (48, 48))
            return Image.fromarray(face_resized)
        except Exception as e:
            print(f"[MediaPipe Error] Resize/Crop failed: {e}")
            return None
