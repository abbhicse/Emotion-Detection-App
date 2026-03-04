# ===============================================================
# ConvNeXt Model for Emotion Detection with Checkpointing & Resume
# ===============================================================
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import timm

# ============================================
# Setup
# ============================================
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ============================================
# Transforms
# ============================================
transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

transform_val = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

train_dir = '/content/drive/MyDrive/Colab Notebooks/data/train'
val_dir = '/content/drive/MyDrive/Colab Notebooks/data/test'

train_loader = DataLoader(datasets.ImageFolder(train_dir, transform=transform_train),
                          batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(datasets.ImageFolder(val_dir, transform=transform_val),
                        batch_size=16, shuffle=False, num_workers=2)

print(f"Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}")

# ============================================
# Build ConvNeXt Model
# ============================================
def build_model():
    model = timm.create_model('convnextv2_tiny', pretrained=True)
    print("Original classifier:", model.get_classifier())
    model.reset_classifier(num_classes=len(EMOTION_CLASSES))
    return model.to(device)

# ============================================
# Training Function (with Resume & Checkpoints)
# ============================================
def train_model(model, epochs=30, lr=1e-4, accum_steps=4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    checkpoint_dir = "/content/drive/MyDrive/Colab Notebooks/models/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # === Resume from last checkpoint if available ===
    start_epoch = 0
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")])
    if checkpoint_files:
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming from checkpoint: {latest_checkpoint} (epoch {start_epoch})")
    else:
        print("No checkpoint found — starting from scratch")

    train_losses, val_losses = [], []

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels) / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * images.size(0) * accum_steps

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # === Validation ===
        model.eval()
        val_loss = 0.0
        correct = 0

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

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {accuracy:.4f}")

        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1:02d}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'accuracy': accuracy
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    # Final full model save
    final_model_path = "/content/drive/MyDrive/Colab Notebooks/models/convnext_emotion.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")

    plot_loss(train_losses, val_losses)

# ============================================
# Plot Training Curve
# ============================================
def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ConvNeXt Training Curve")
    plt.legend()
    plt.grid(True)
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/convnext_loss.png")
    plt.show()

# ============================================
# Main
# ============================================
if __name__ == "__main__":
    model = build_model()
    train_model(model, epochs=30, lr=1e-4, accum_steps=4)
