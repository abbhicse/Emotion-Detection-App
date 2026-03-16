import streamlit as st
from utils.image_utils import validate_image, load_image
import importlib

st.set_page_config(page_title="Emotion Detection App", layout="centered")
st.title("Emotion Detection from Images")
st.markdown("Upload a clear image of a human face, and the model will predict the emotion expressed.")

model_options = {
    "Custom CNN (emotion_cnn.pt)": {
        "path": "models/emotion_cnn.pt",
        "module": "utils.emotion_classifier_cnn"
    },
    "ResNet50 (resnet50_emotion.pt)": {
        "path": "models/resnet50_emotion.pt",
        "module": "utils.emotion_classifier_resnet"
    },
    "ConvNeXt V2 Tiny (convnext_emotion.pt)": {
        "path": "models/convnext_emotion.pt",
        "module": "utils.emotion_classifier_convnext"
    }
}

selected_model_name = st.selectbox("Select Emotion Detection Model", list(model_options.keys()))
selected_model_info = model_options[selected_model_name]


@st.cache_resource
def get_model_and_predictor(model_path, module_name):
    model_module = importlib.import_module(module_name)
    model = model_module.load_model(model_path)
    return model, model_module.predict_emotion


def get_helpers():
    face_module = importlib.import_module("utils.face_detection")
    feature_module = importlib.import_module("utils.feature_extraction")
    return face_module.detect_face, feature_module.extract_landmarks


try:
    model, predict_emotion = get_model_and_predictor(
        selected_model_info["path"],
        selected_model_info["module"]
    )
    detect_face, extract_landmarks = get_helpers()
except Exception as e:
    st.error(f"Startup error: {e}")
    st.stop()


uploaded_file = st.file_uploader("Upload an Image (JPG/PNG only)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if validate_image(uploaded_file):
        image = load_image(uploaded_file)
        if image is not None:
            face = detect_face(image)
            if face is not None:
                try:
                    _ = extract_landmarks(face)
                except Exception:
                    pass

                prediction = predict_emotion(model, face)
                st.success(f"Predicted Emotion using **{selected_model_name}**: {prediction}")
            else:
                st.error("No face detected. Please upload a clear frontal face image.")
    else:
        st.warning("Invalid image. Please upload a valid JPG/PNG file under 5MB.")
