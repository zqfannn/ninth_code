import numpy as np
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, accuracy_score
from PIL import Image
from torch.utils.data import Dataset
from skimage.morphology import binary_erosion


class SegmentationDataset(Dataset):
    def __init__(self, image_folder, label_folder, transform=None):
        self.images = sorted(glob.glob(os.path.join(image_folder, '*.png')))
        self.labels = sorted(glob.glob(os.path.join(label_folder, '*.png')))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        label = Image.open(self.labels[index]).convert('L')

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
            label = label.squeeze(0)

        return image, label.long()


# Set the image and label folders
image_folder = 'MNIST'
label_folder = 'MNIST'

# Define data preprocessing transforms
transform = transforms.Compose([
    transforms.ToTensor()
])

# Create dataset instance
dataset = SegmentationDataset(image_folder, label_folder, transform=transform)

# Create data loader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

# Define the UNet model
class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

# Create an instance of the UNet model
model = UNet(num_classes=10)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define loss function
criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore the label 255

# Define the training function
def train(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for i, (images, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # if (i + 1) % 10 == 0:  # Print the loss every 10 steps
        #     print(f"Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)

# Define the evaluation function
def evaluate(model, dataloader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.flatten().tolist())
            y_pred.extend(predicted.flatten().tolist())

    # Calculate Pixel Accuracy
    pixel_accuracy = accuracy_score(y_true, y_pred)

    # Calculate mIoU
    conf_matrix = confusion_matrix(y_true, y_pred)
    intersection = np.diag(conf_matrix)
    union = np.sum(conf_matrix, axis=1) + np.sum(conf_matrix, axis=0) - intersection
    iou = intersection / union.astype(np.float32)
    mIoU = np.mean(iou)

    # Calculate Boundary IoU
    boundary_iou = boundary_iou_score(y_true, y_pred)

    return pixel_accuracy, mIoU, boundary_iou

# Define the Boundary IoU evaluation function
def boundary_iou_score(y_true, y_pred):
    border_true = extract_boundary(y_true)
    border_pred = extract_boundary(y_pred)
    intersection = np.logical_and(border_true, border_pred)
    union = np.logical_or(border_true, border_pred)
    iou = np.sum(intersection) / np.sum(union)
    return iou

# Define the boundary extraction function
def extract_boundary(mask):
    mask = np.array(mask)
    mask = mask.squeeze()
    mask = mask.astype(bool)
    border = mask ^ binary_erosion(mask)
    return border

# Train the model
for epoch in range(10):
    print(f"Epoch [{epoch+1}/{10}]")
    train_loss = train(model, dataloader, criterion, optimizer)
    pixel_accuracy, mIoU, boundary_iou = evaluate(model, dataloader)
    output_str = (
        f"Train Loss: {train_loss:.4f}, "
        f"Pixel Accuracy: {pixel_accuracy:.4f}, "
        f"mIoU: {mIoU:.4f}, "
        f"Boundary IoU: {boundary_iou:.4f}"
    )
    print(output_str)
