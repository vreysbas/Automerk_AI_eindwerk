import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os

# 1. Instellingen
DATA_DIR = "dataset_medium"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model():
    # 2. Data Transformaties (ResNet verwacht 224x224)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Data Inladen
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Klassen gevonden: {train_dataset.classes}")

    # 4. Het Model (ResNet18 - Transfer Learning)
    model = models.resnet18(weights='IMAGENET1K_V1')
    # Pas de laatste laag aan naar 3 klassen (Audi, BMW, Mercedes)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)
    model = model.to(DEVICE)

    # 5. Loss en Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. Training Loop
    print(f"Start training op {DEVICE}...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Testen na elke epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] - Loss: {running_loss / len(train_loader):.4f} - Accuracy: {accuracy:.2f}%")

    # 7. Model Opslaan
    torch.save(model.state_dict(), "automerk_model.pth")
    print("Model opgeslagen als automerk_model.pth")


if __name__ == "__main__":
    train_model()