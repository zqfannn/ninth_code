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
            image = self.transform(image)  # 应用图像预处理变换
            label = self.transform(label)  # 应用标签预处理变换
            label = label.squeeze(0)  # 去除标签的通道维度

        return image, label.long()


# Set the image and label folders
image_folder = 'MNIST'
label_folder = 'MNIST'

# 定义数据预处理的变换
transform = transforms.Compose([
    transforms.ToTensor()
])

# 创建数据集实例
dataset = SegmentationDataset(image_folder, label_folder, transform=transform)

# 创建数据加载器
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

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 定义损失函数
criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore the label 255

# Define the training function
def train(model, dataloader, criterion, optimizer):
    model.train()
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

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


# 定义边界交并比 (Boundary IoU) 的评估函数
def boundary_iou_score(y_true, y_pred):
    border_true = extract_boundary(y_true)
    border_pred = extract_boundary(y_pred)
    intersection = np.logical_and(border_true, border_pred)
    union = np.logical_or(border_true, border_pred)
    iou = np.sum(intersection) / np.sum(union)
    return iou

# 定义边界提取函数
def extract_boundary(mask):
    mask = np.array(mask)
    mask = mask.squeeze()
    mask = mask.astype(bool)
    border = mask ^ binary_erosion(mask)
    return border



# Train the model
for epoch in range(10):
    train(model, dataloader, criterion, optimizer)
    pixel_accuracy, mIoU, boundary_iou = evaluate(model, dataloader)
    print(f"Epoch {epoch+1}:")
    print(f"Pixel Accuracy: {pixel_accuracy:.4f}")
    print(f"mIoU: {mIoU:.4f}")
    print(f"Boundary IoU: {boundary_iou:.4f}")
