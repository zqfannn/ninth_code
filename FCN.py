import numpy as np
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, confusion_matrix

# 自定义数据集类
class SegmentationDataset(Dataset):
    def __init__(self, image_folder, label_folder, transform=None):
        # 获取所有图像文件的路径
        self.images = sorted(glob.glob(os.path.join(image_folder, '*.png')))
        # 获取所有标签文件的路径
        self.labels = sorted(glob.glob(os.path.join(label_folder, '*.png')))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
         # 读取图像并转换为RGB格式
        image = Image.open(self.images[index]).convert('RGB')
         # 读取标签并转换为灰度图像
        label = Image.open(self.labels[index]).convert('L')

        if self.transform:
            # 应用图像转换
            image = self.transform(image)
            label = self.transform(label)
            # 去除维度为1的维度并转换为长整型
            label = label.squeeze(0).long()

        return image, label

image_folder = 'MNIST'
label_folder = 'MNIST'

transform = transforms.Compose([
    transforms.ToTensor()    # 将图像转换为张量
])

dataset = SegmentationDataset(image_folder, label_folder, transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

# FCN模型定义
class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


# 创建FCN模型实例
model = FCN(num_classes=10)

params = list(model.parameters())
if len(params) == 0:
    raise ValueError("The model has no learnable parameters")

# 定义损失函数
criterion = nn.CrossEntropyLoss()
# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train(model, dataloader, criterion, optimizer):
    model.train()
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 计算性能指标
def calculate_performance_metrics(y_true, y_pred, num_classes):
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

     # 排除未知类别的计算
    unknown_indices = np.where(y_true == num_classes)[0]
    y_true = np.delete(y_true, unknown_indices)
    y_pred = np.delete(y_pred, unknown_indices)

    # 计算指标
    pixel_accuracy = accuracy_score(y_true, y_pred)
    conf_mat = confusion_matrix(y_true, y_pred)
    iou = np.diag(conf_mat) / (np.sum(conf_mat, axis=1) + np.sum(conf_mat, axis=0) - np.diag(conf_mat) + 1e-6)
    mean_iou = np.mean(iou)
    boundary_iou = iou.sum() / (num_classes - 1)

    return pixel_accuracy, mean_iou, boundary_iou

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = 0.0
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)

    train_loss = train_loss / len(dataloader.dataset)
    
    # 获取所有样本的真实标签和预测标签
    y_true_all = []
    y_pred_all = []
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true_all.append(labels.numpy())
            y_pred_all.append(predicted.numpy())
    
    # 处理预测结果
    y_true_all = np.concatenate(y_true_all, axis=0)
    y_pred_all = np.concatenate(y_pred_all, axis=0)

    # 进行性能指标计算
    pixel_accuracy, mean_iou, boundary_iou = calculate_performance_metrics(y_true_all, y_pred_all, num_classes=10)
    print("Epoch [{}/{}], Train Loss: {:.4f}, Pixel Accuracy: {:.4f}, mIoU: {:.4f}, Boundary IoU: {:.4f}".format(
    epoch+1, num_epochs, train_loss, pixel_accuracy, mean_iou, boundary_iou))

