from simple_deep_learning.mnist_extended.semantic_segmentation import create_semantic_segmentation_dataset
import tensorflow as tf
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from simple_deep_learning.mnist_extended.semantic_segmentation import display_grayscale_array, plot_class_masks
from sklearn.metrics import accuracy_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

from simple_deep_learning.mnist_extended.semantic_segmentation import display_segmented_image

import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Activation
import cv2

import os

def unet():
    # Input
    inputs = Input(shape=(60, 60, 1))

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bottom
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)

    # Decoder
    up4 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv3))
    merge4 = concatenate([conv2, up4], axis=3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(merge4)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)

    up5 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv4))
    merge5 = concatenate([conv1, up5], axis=3)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(merge5)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5)

    # Output
    outputs = Conv2D(3, 1, activation='softmax')(conv5)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Create U-Net model


# Create U-Net model

def segnet():
    # Encoder
    inputs = tf.keras.Input(shape=(60, 60, 1))
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    pool1 = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    pool2 = MaxPooling2D(pool_size=(2, 2))(x)

    # Decoder
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    up1 = UpSampling2D(size=(2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    up2 = UpSampling2D(size=(2, 2))(x)

    # Output
    outputs = Conv2D(3, (1, 1), activation='softmax', padding='valid')(up2)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    return model

# def fcn_network(input_shape):
#     inputs = tf.keras.Input(shape=input_shape)
    
#     # Encoder
#     conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
#     conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
#     pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
#     conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
#     conv4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
#     pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    
#     conv5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
#     conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)
#     pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv6)
    
#     # Decoder
#     conv7 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
#     conv8 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv7)
#     up1 = tf.keras.layers.UpSampling2D((2, 2))(conv8)
    
#     conv9 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up1)
#     conv10 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv9)
#     up2 = tf.keras.layers.UpSampling2D((2, 2))(conv10)
    
#     conv11 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up2)
#     conv12 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv11)
#     up3 = tf.keras.layers.UpSampling2D((2, 2))(conv12)
    
#     conv13 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up3)
#     conv14 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv13)
    
#     # Output
#     outputs = tf.keras.layers.Conv2D(3, (1, 1), activation='softmax')(conv14)
    
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     return model



np.random.seed(1)
train_x, train_y, test_x, test_y = create_semantic_segmentation_dataset(num_train_samples=1000,
                                                                        num_test_samples=200,
                                                                        image_shape=(60, 60),
                                                                        max_num_digits_per_image=4,
                                                                        num_classes=3)

print(train_x.shape, train_y.shape)

# i = np.random.randint(len(train_x))

# display_grayscale_array(array=train_x[i])

# plot_class_masks(train_y[i])

# tf.keras.backend.clear_session()

def pixel_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def iou(y_true, y_pred):
    confusion_mat = confusion_matrix(y_true, y_pred)
    return np.diag(confusion_mat) / (np.sum(confusion_mat, axis=1) + np.sum(confusion_mat, axis=0) - np.diag(confusion_mat))

def mean_iou(y_true, y_pred):
    confusion_mat = confusion_matrix(y_true, y_pred)
    iou = np.diag(confusion_mat) / (np.sum(confusion_mat, axis=1) + np.sum(confusion_mat, axis=0) - np.diag(confusion_mat))
    return np.mean(iou)




# def calculate_performance_metrics(y_true, y_pred, num_classes):
#     pixel_accuracy = accuracy_score(y_true, y_pred)
#     confusion_mat = confusion_matrix(y_true, y_pred)
#     iou = np.diag(confusion_mat) / (np.sum(confusion_mat, axis=1) + np.sum(confusion_mat, axis=0) - np.diag(confusion_mat))
#     mean_iou = np.mean(iou)
#     boundary_iou = np.sum(iou) / (num_classes + 1)
#     return pixel_accuracy, mean_iou, boundary_iou


# model = models.Sequential()
# model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=train_x.shape[1:], padding='same'))
# model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(layers.Conv2D(filters=train_y.shape[-1], kernel_size=(3, 3), activation='sigmoid', padding='same'))



# Create SegNet model
model = segnet()
# model = unet()
# 创建 FCN 网络模型

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                       tf.keras.metrics.Recall(),
                       tf.keras.metrics.Precision()])

history = model.fit(train_x, train_y, epochs=5,
                    validation_data=(test_x, test_y))


# checkpoint_path = "weight.ckpt"
# model.save_weights(checkpoint_path)

test_y_predicted = model.predict(test_x)





# np.random.seed(6)
# for _ in range(3):
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
#     i = np.random.randint(len(test_y_predicted))
#     print(f'Example {i}')
#     display_grayscale_array(test_x[i], ax=ax1, title='Input image')
#     display_segmented_image(test_y_predicted[i], ax=ax2, title='Segmented image', threshold=0.5)
#     plot_class_masks(test_y[i], test_y_predicted[i], title='y target and y predicted sliced along the channel axis')


def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w, c  = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2) # 计算图像对角线长度
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
        
    mask = mask.astype(np.uint8)
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    
    # 因为之前向四周填充了0, 故而这里不再需要四周
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    
    # G_d intersects G in the paper.
    return mask - mask_erode



def boundary_iou(gt, dt, dilation_ratio=0.005, cls_num=3):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    # 注意 gt 和 dt 的 shape 不一样
    # gt = gt[0, 0]
    # dt = dt[0]
    
    # 这里为了让 gt 和 dt 变为 (h, w) 而不是 (1, h, w) 或者 (1, 1, h, w)
    
	# 注意这里的类别转换主要是为了后边计算边界
    # gt = gt.numpy().astype(np.uint8)
    # dt = dt.numpy().astype(np.uint8)
    
    gt = gt.astype(np.uint8)
    dt = dt.astype(np.uint8)
    
    boundary_iou_list = []
    for i in range(cls_num):
        
        gt_i = (gt == i)
        dt_i = (dt == i)
        
        gt_boundary = mask_to_boundary(gt_i, dilation_ratio)
        dt_boundary = mask_to_boundary(dt_i, dilation_ratio)
        intersection = ((gt_boundary * dt_boundary) > 0).sum()
        union = ((gt_boundary + dt_boundary) > 0).sum()
        if union < 1:
            boundary_iou_list.append(0)
            continue
        boundary_iou = intersection / union
        boundary_iou_list.append(boundary_iou)  
        
    return np.sum(boundary_iou_list)

def calculate_iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask)
    union = np.logical_or(pred_mask, true_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def calculate_miou(predictions, labels):
    num_classes = predictions.shape[2]
    iou_sum = 0.0
    for class_id in range(num_classes):
        pred_mask = predictions[:,:,class_id]
        true_mask = labels[:,:,class_id]
        pred_mask = (pred_mask>0.0001)
        true_mask = ( true_mask>0.0001)
        iou = calculate_iou(pred_mask, true_mask)
        iou_sum += iou
    # miou = iou_sum / num_classes
    miou = iou_sum 
    return miou




b_iou=0
# BinaryAccuracy = 0
m_iou = 0

for i in range(200):
    b_iou = b_iou + boundary_iou(test_y[i], test_y_predicted[i], dilation_ratio=0.005, cls_num=3)
    # BinaryAccuracy = BinaryAccuracy + tf.keras.metrics.BinaryAccuracy(test_y[i],test_y_predicted[i])
    m_iou = m_iou + calculate_miou(test_y[i], test_y_predicted[i])
 
b_iou = b_iou/200
# BinaryAccuracy = BinaryAccuracy/200
m_iou = m_iou/200

print('boudary_iou:',b_iou,'m_iou:',m_iou)

for i in range(0,200,50):
    label = test_y[i]
    pred = test_y_predicted[i]
    plt.imshow(label)
    s=str(i)
    #plt.savefig(os.path.join('./result/FCN', s + '_label.png'))
    plt.savefig(os.path.join('./result/segnet', s + '_label.png'))
    plt.imshow(pred)
    #plt.savefig(os.path.join('./result/FCN', s + '_pred.png'))
    plt.savefig(os.path.join('./result/segnet', s + '_pred.png'))


















