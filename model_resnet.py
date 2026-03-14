# model_resnet50.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# =========================
# 0. Setup
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# =========================
# 1. Data Transforms & Loading
# =========================

transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),   # FER-2013 is grayscale → make 3-channel RGB
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

transform_val = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

train_dir = '/content/drive/MyDrive/Colab Notebooks/data/train'
val_dir = '/content/drive/MyDrive/Colab Notebooks/data/test'

train_data = datasets.ImageFolder(train_dir, transform=transform_train)
val_data = datasets.ImageFolder(val_dir, transform=transform_val)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=2)

print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

# =========================
# 2. Build Model (ResNet50)
# =========================

def build_model():
    # Load pretrained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Input is already 3-channel RGB, so no need to modify first conv
    # Modify the final FC layer to match emotion classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(EMOTION_CLASSES))

    return model.to(device)

# =========================
# 3. Training Function
# =========================

def train_model(model, epochs=30, lr=3e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss, correct = 0.0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        accuracy = correct / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {accuracy:.4f}")

    # Save trained model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/resnet50_emotion.pt")
    print("Model saved to: models/resnet50_emotion.pt")

    plot_loss(train_losses, val_losses)

# =========================
# 4. Plot Loss Function
# =========================

def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss (ResNet50)")
    plt.legend()
    plt.grid(True)
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/loss_curve_resnet50.png")
    plt.show()

# =========================
# 5. Run Training
# =========================

if __name__ == '__main__':
    model = build_model()
    train_model(model)
