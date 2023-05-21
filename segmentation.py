import numpy as np
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from PIL import Image
from torch.utils.data import Dataset

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


# 设置数据集文件夹路径
image_folder = 'MNIST'
label_folder = 'MNIST'

# 定义数据预处理的转换器
transform = transforms.Compose([
    transforms.ToTensor()
])

# 创建数据集实例
dataset = SegmentationDataset(image_folder, label_folder, transform=transform)

# 创建数据加载器
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

# 定义UNet模型
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

# 创建UNet模型实例
model = UNet(num_classes=10)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 定义训练函数
def train(model, dataloader, criterion, optimizer):
    model.train()
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 创建训练数据集实例
train_dataset = SegmentationDataset(image_folder, label_folder, transform=transform)

# 创建测试数据集实例
test_dataset = SegmentationDataset(image_folder, label_folder, transform=transform)

# 创建数据加载器
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 创建模型实例
model = UNet(num_classes=10)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = 0.0
    model.train()
    for images, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_dataset)

    # 在每个训练周期结束时进行评估
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in test_dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    # 过滤掉未知标签
    valid_indices = [i for i, label in enumerate(y_true) if label != "unknown"]
    y_true_valid = [y_true[i] for i in valid_indices]
    y_pred_valid = [y_pred[i] for i in valid_indices]

    # 扁平化y_true_valid和y_pred_valid
    y_true_valid_flat = np.concatenate(y_true_valid)
    y_pred_valid_flat = np.concatenate(y_pred_valid)

    # 计算像素准确率
    pixel_accuracy = accuracy_score(y_true_valid_flat, y_pred_valid_flat)

    # 打印训练损失和评估指标
    print("Epoch [{}/{}], Train Loss: {:.4f}, Pixel Accuracy: {:.4f}".format(epoch+1, num_epochs, train_loss, pixel_accuracy))
