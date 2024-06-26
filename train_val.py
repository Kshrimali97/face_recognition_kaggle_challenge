import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import pandas as pd
from PIL import Image
import os

df = pd.read_csv(os.path.join(os.getcwd(), "train.csv"), index_col=False)


class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.csv = pd.read_csv(csv_file, index_col=False)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, str(self.csv.iloc[index, 0]) + ".jpg")
        image = Image.open(img_path)
        label = self.csv.iloc[index, 2]
        if self.transform:
            image = self.transform(image)
        return (image, label)


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        # transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15, resample=False, fillcolor=0),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

dataset = CustomDataset(csv_file="train.csv", root_dir="train", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


num_classes = 100
model = models.vgg16(pretrained=True)
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, num_classes)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    training_loss: float = 0.0
    for batch_idx, (data, targets) in enumerate(train_loader):
        data.to(device)
        targets.to(device)

        outputs = model(data)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss += loss.item()

        if batch_idx % 100 == 0:
            print(
                f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}"
            )
    training_loss /= len(train_loader)

    model.eval()
    val_loss: float = 0.0
    correct_predictions: int = 0
    total_predictions: int = 0
    with torch.no_grad():
        for data, targets in val_loader:
            data.to(device)
            targets.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            _, predictions = torch.max(outputs, 1)
            correct_predictions += (predictions == targets).sum()
            total_predictions += predictions.size(0)

    val_loss /= len(val_loader)
    accuracy = correct_predictions / total_predictions

    print(
        f"Epoch {epoch+1}/{num_epochs}, Training Loss: {training_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.4f}"
    )

torch.save(model.state_dict(), "facial_recognition_model_kanika.pth")

model.eval()
predictions_val = []
with torch.no_grad():
    for data, targets in val_loader:
        data.to(device)
        targets.to(device)

        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        predictions_val.append(predicted)

predictions_val = torch.cat(predictions_val, 0)
predictions_df_val = pd.DataFrame(
    {"ID": val_dataset.dataset.csv.iloc[:, 0], "Category": predictions_val}
)
predictions_df_val.to_csv("predictions_val.csv", index=False)


# load the testing dataset (we create a dummy test.csv to basically mimic the train.csv file so as to ease dataloading)
test_dataset = CustomDataset(csv_file="test.csv", root_dir="test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# output the predictions on the test set
model.eval()
predictions = []
with torch.no_grad():
    for data, targets in test_loader:
        data.to(device)
        targets.to(device)

        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        predictions.append(predicted)

predictions = torch.cat(predictions, 0)
predictions_df = pd.DataFrame(
    {"ID": test_dataset.csv.iloc[:, 0], "Category": predictions}
)
predictions_df.to_csv("kanika_kaggle_submission_1.csv", index=False)
