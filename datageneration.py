import numpy as np
import os
from torchvision.datasets import MNIST
from torchvision.transforms import ToPILImage

# 生成语义分割数据集
mnist = MNIST('MNIST', download=True, train=True)
images = mnist.data.numpy()
labels = mnist.targets.numpy()

# 划分训练集和测试集
train_images = images[:50000]
train_labels = labels[:50000]
test_images = images[50000:]
test_labels = labels[50000:]

# 导出为图像文件
output_dir = 'MNIST'

# 创建保存图像的目录
os.makedirs(output_dir, exist_ok=True)

# 保存训练集图像
for i, image in enumerate(train_images):
    image_pil = ToPILImage()(image)
    image_pil.save(os.path.join(output_dir, f'train_{i}.png'))

# 保存测试集图像
for i, image in enumerate(test_images):
    image_pil = ToPILImage()(image)
    image_pil.save(os.path.join(output_dir, f'test_{i}.png'))
