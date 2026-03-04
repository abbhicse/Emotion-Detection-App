import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
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
# 1. ENHANCED CNN MODEL
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

    def _initialize_weights(self):  # Custom weight init
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
# 2. DATA PREPROCESSING
# =========================
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dir = '/content/drive/MyDrive/Colab Notebooks/data/train'
val_dir = '/content/drive/MyDrive/Colab Notebooks/data/test'

print("Loading dataset...")
train_data = datasets.ImageFolder(train_dir, transform=transform)
val_data = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=2)

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")


# =========================
# 3. TRAINING LOOP
# =========================
def train_model(model, epochs=50, learning_rate=0.0005):
    print("Training started...\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_list = []
    val_loss_list = []

    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_loss_list.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_loss_list.append(val_loss)
        accuracy = correct / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - "
              f"Val Loss: {val_loss:.4f} - Val Acc: {accuracy:.4f}")

        # Optional Early Stopping (commented for now)
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save(model.state_dict(), "models/best_emotion_cnn.pt")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time / 60:.2f} minutes")

    # Save final model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/emotion_cnn.pt")
    print("Model saved to: models/emotion_cnn.pt")

    plot_loss(train_loss_list, val_loss_list)


# =========================
# 4. PLOT LOSS CURVES
# =========================
def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/loss_curve.png")
    plt.show()
    print("Loss curve saved as plots/loss_curve.png")


# =========================
# 5. RUN
# =========================
if __name__ == "__main__":
    model = EmotionCNN().to(device)
    train_model(model)
