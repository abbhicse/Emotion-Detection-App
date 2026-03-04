# utils/emotion_classifier_convnext.py

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm

# Emotion labels
EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================
# 1. Load ConvNeXt model
# ============================================

def load_model(path='models/convnext_emotion.pt'):
    """
    Load the fine-tuned ConvNeXtV2 model trained on FER dataset.
    """

    # ✅ Use the SAME architecture you trained with
    model = timm.create_model('convnextv2_tiny', pretrained=False, num_classes=len(EMOTION_CLASSES))

    # ✅ Load model weights safely (ignore mismatched keys)
    checkpoint = torch.load(path, map_location=device)
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)

    if missing or unexpected:
        print(f"⚠️ Warning: Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    model = model.to(device)
    model.eval()
    print("✅ ConvNeXtV2 model loaded successfully.")
    return model


# ============================================
# 2. Predict emotion from PIL face image
# ============================================

def predict_emotion(model, face_img: Image.Image):
    """
    Predict emotion using ConvNeXt on a cropped face image.

    Args:
        model: Loaded ConvNeXt model
        face_img: PIL image (RGB or grayscale)

    Returns:
        str: predicted emotion
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    input_tensor = transform(face_img).unsqueeze(0).to(device)  # Shape: [1, 3, 224, 224]

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        return EMOTION_CLASSES[pred_idx]
