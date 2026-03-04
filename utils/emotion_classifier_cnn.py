# utils/emotion_classifier_cnn.py

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# =========================
# 0. Setup
# =========================

# Emotion class labels
EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 1. Define the EmotionCNN model
# =========================

class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling

        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(EMOTION_CLASSES))
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.gap(x)
        x = self.fc_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

# =========================
# 2. Load the trained model
# =========================

def load_model(path='models/emotion_cnn.pt'):
    """
    Load the trained EmotionCNN model.
    """
    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

# =========================
# 3. Predict emotion from a face image
# =========================

def predict_emotion(model, face_img: Image.Image):
    """
    Predict the emotion from a single face image using EmotionCNN.

    Args:
        model: Loaded EmotionCNN model.
        face_img: PIL.Image of a cropped face (grayscale or RGB).

    Returns:
        Predicted emotion label (str).
    """
    transform = transforms.Compose([
        transforms.Grayscale(),                   # Convert to 1 channel
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Prepare input tensor
    input_tensor = transform(face_img).unsqueeze(0).to(device)  # Shape: [1, 1, 48, 48]

    with torch.no_grad():
        output = model(input_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()
        return EMOTION_CLASSES[predicted_idx]
