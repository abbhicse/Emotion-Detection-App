# utils/emotion_classifier_resnet.py

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Emotion classes (same as training)
EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path='models/resnet50_emotion.pt'):
    """
    Load the fine-tuned ResNet50 model for emotion classification.
    Assumes the model was trained with torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
    """
    # Load pretrained ResNet50 and modify final layer
    model = models.resnet50(weights=None)  # We are loading weights manually
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(EMOTION_CLASSES))

    # Load trained weights
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()

    return model


def predict_emotion(model, face_img: Image.Image):
    """
    Predict emotion using a 48x48 RGB face image (PIL.Image).
    The input should already be a cropped face.
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Match training
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    input_tensor = transform(face_img).unsqueeze(0).to(device)  # Shape: [1, 3, 48, 48]

    with torch.no_grad():
        output = model(input_tensor)
        predicted_idx = output.argmax(dim=1).item()
        return EMOTION_CLASSES[predicted_idx]
