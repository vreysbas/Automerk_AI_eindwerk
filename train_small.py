import os
import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "carbrand_model.pth"
FORCE_TRAIN = "--train" in sys.argv  # opnieuw trainen als je --train meegeeft

# ----------------------------
# Data transforms
# ----------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----------------------------
# Datasets & loaders
# ----------------------------
train_data = datasets.ImageFolder("dataset_small/train", transform=train_transform)
test_data  = datasets.ImageFolder("dataset_small/test",  transform=test_transform)

train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=4, shuffle=False)

# ----------------------------
# Model
# ----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 224 -> conv/pool -> ~ 16 x 54 x 54 (met jouw huidige layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 54 * 54, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN(num_classes=len(train_data.classes))

# ----------------------------
# Load or Train
# ----------------------------
if os.path.exists(MODEL_PATH) and not FORCE_TRAIN:
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    saved_classes = checkpoint["classes"]

    # kleine safety check: klassenvolgorde moet overeenkomen
    if saved_classes != train_data.classes:
        print("⚠️ Waarschuwing: de opgeslagen klassenvolgorde is anders dan je huidige dataset.")
        print("Opgeslagen:", saved_classes)
        print("Huidig   :", train_data.classes)
        print("Train opnieuw met: python train_small.py --train")
    else:
        print(f"✅ Model geladen uit {MODEL_PATH} (training overgeslagen)")
else:
    print("🧠 Geen opgeslagen model gevonden (of --train gebruikt). Training start...")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    epochs = 15
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for imgs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss:.4f}")

    # Opslaan (model "onthoudt" training)
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": train_data.classes
    }, MODEL_PATH)

    print(f"✅ Model opgeslagen als {MODEL_PATH}")

# ----------------------------
# Evaluate (accuracy)
# ----------------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in test_loader:
        outputs = model(imgs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

acc = 100 * correct / total
print(f"\nTest accuracy: {acc:.2f}%")
print("Klassenvolgorde:", train_data.classes)

# ----------------------------
# Error analysis
# ----------------------------
print("\n--- FOUTENANALYSE OP TESTSET ---")
with torch.no_grad():
    for imgs, labels in test_loader:
        outputs = model(imgs)
        _, predicted = torch.max(outputs, 1)

        for i in range(len(labels)):
            if predicted[i] != labels[i]:
                voorspeld = train_data.classes[predicted[i].item()]
                correct_merk = train_data.classes[labels[i].item()]
                print(f"FOUT: voorspeld={voorspeld} | correct={correct_merk}")
