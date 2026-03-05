from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(
    "dataset_small/train",
    transform=transform
)

print("Gevonden merken:", train_data.classes)
print("Aantal trainingsbeelden:", len(train_data))

loader = DataLoader(train_data, batch_size=4, shuffle=True)
images, labels = next(iter(loader))

print("Batch shape:", images.shape)
print("Labels:", labels)