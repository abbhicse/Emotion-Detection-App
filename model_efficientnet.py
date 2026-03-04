import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# =========================
# 0. DEVICE SETUP
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Emotion classes
EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# =========================
# 1. DATA TRANSFORMS
# =========================
transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # FER-2013 is grayscale
    transforms.Resize((224, 224)),                # Required for EfficientNet
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transform_val = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dir = '/content/drive/MyDrive/Colab Notebooks/data/train'
val_dir = '/content/drive/MyDrive/Colab Notebooks/data/test'

train_data = datasets.ImageFolder(train_dir, transform=transform_train)
val_data = datasets.ImageFolder(val_dir, transform=transform_val)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=2)

print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

# =========================
# 2. BUILD MODEL
# =========================
def build_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Replace classifier for 7 emotions
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(EMOTION_CLASSES))

    return model.to(device)

# =========================
# 3. TRAINING FUNCTION
# =========================
def train_model(model, epochs=30, lr=0.0003):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)

        train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0
        correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

        val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        accuracy = correct / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Accuracy: {accuracy:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/efficientnet_emotion.pt")
    print("Model saved to: models/efficientnet_emotion.pt")

    plot_loss(train_losses, val_losses)

# =========================
# 4. PLOT LOSS
# =========================
def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/efficientnet_loss.png")
    plt.show()

# =========================
# 5. RUN
# =========================
if __name__ == "__main__":
    model = build_model()
    train_model(model)
