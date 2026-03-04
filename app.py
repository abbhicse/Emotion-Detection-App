# app.py
import streamlit as st
from utils.image_utils import validate_image, load_image
from utils.face_detection import detect_face
from utils.feature_extraction import extract_landmarks
import importlib

# Page settings
st.set_page_config(page_title="Emotion Detection App", layout="centered")
st.title("Emotion Detection from Images")
st.markdown("Upload a clear image of a human face, and the model will predict the emotion expressed.")

# === Dropdown: Model Selection ===
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

# === Dynamic Module Import ===
@st.cache_resource
def get_model_and_predictor(model_path, module_name):
    model_module = importlib.import_module(module_name)
    model = model_module.load_model(model_path)
    return model, model_module.predict_emotion

model, predict_emotion = get_model_and_predictor(
    selected_model_info["path"],
    selected_model_info["module"]
)

# === File Upload ===
uploaded_file = st.file_uploader("Upload an Image (JPG/PNG only)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if validate_image(uploaded_file):
        image = load_image(uploaded_file)
        if image is not None:
            #st.image(image, caption="Uploaded Image", use_column_width=True)
            #st.write(f"Image mode: {image.mode}")

            face = detect_face(image)
            if face is not None:
                landmarks = extract_landmarks(face)  # optional

                # === Predict Emotion ===
                prediction = predict_emotion(model, face)

                #st.image(face, caption="Detected Face", use_column_width=True)
                st.success(f"Predicted Emotion using **{selected_model_name}**: {prediction}")
            else:
                st.error("No face detected. Please upload a clear frontal face image.")
    else:
        st.warning("Invalid image. Please upload a valid JPG/PNG file under 5MB.")
