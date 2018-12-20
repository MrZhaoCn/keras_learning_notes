import numpy as  np
import pandas as pd
import os
import glob
import math
import keras
from scipy.misc import imread, imsave, imresize
from natsort import natsorted

path = '../data/chars74'
img_height, img_width = 32, 32

#处理后的图像存储目录
suffix = 'preproc'
trainDataPath = path + "/train" + suffix
testDataPath = path + "/test" + suffix

if not os.path.exists(trainDataPath):
    os.makedirs(trainDataPath)

if not os.path.exists(testDataPath):
    os.makedirs(testDataPath)

#图像大小和色彩的处理
# 图像的色彩(Image Color)
# 训练和测试资料集中几乎所有图像都是彩色图像。预处理的第一步是将所有图像转换为灰阶。它简化了输入到网络的数据，也让模型更能够一般化(generalize)，因为一个蓝色的字母与一个红色字母在这个图像的分类问题上都是相同的。因此把图像颜色的通道(channel)进行缩减的这个预处理应该对最终的准确性没有负面影响，因为大多数文本与背景具有高度对比。
#
# 图像大小的修改(Image Resizing)
# 由于图像具有不同的形状和大小，因此我们必须对图像进行归一化(normalize)处理以便可以决定模型的输入。这个处理会有两个主要的问题需要解决：我们选择哪种图像大小(size)？我们是否该保持图像宽高比(aspect ratio)?
#
# 起初，我也认为保持图像的高宽比会更好，因为它不会任意扭曲图像。这也可能导致O和O（大写o和零）之间的混淆。不过，经过一番测试，似乎没有保持宽高比的模型效果更好。
#
# 关于图像尺寸，16×16的图像允许非常快速的训练，但不能给出最好的结果。这些小图像是快速测试想法的完美选择。使用32×32的图像使训练相当快，并提供良好的准确性。最后，与32×32图像相比，使用64×64图像使得训练相当缓慢并略微提高了结果。我选择使用32×32的图像，因为它是速度和准确性之间的最佳折衷。

for datasetType in ["train","test"]:
    imgFiles = natsorted(glob.glob(path + "/" + datasetType + "/*"))
    imgData = np.zeros((len(imgFiles), img_height, img_width))
    for i, imgFilePath in enumerate(imgFiles):
        # True:代表读取图像时顺便将多阶图像,打平成灰阶(单一通道:one channel)
        img = imread(imgFilePath, True)
        imgResized = imresize(img, (img_height, img_width))
        imgData[i] = imgResized

        # 将修改的图像储存到档案系统(方便视觉化了解)
        filename = os.path.basename(imgFilePath)
        filenameDotSplit = filename.split(".")
        newFilename = str(int(filenameDotSplit[0])).zfill(5) + "." + filenameDotSplit[- 1].lower()
        newFilepath = path + "/" + datasetType + suffix + "/" + newFilename
        imsave(newFilepath, imgResized)

    # 新增加"Channel"的维度
    print("Before: ", imgData.shape)
    imgData = imgData[:, :, :, np.newaxis]  # chanel  维度为1
    print("After: ", imgData.shape)

    # 进行资料(pixel值)标准化
    imgData = imgData.astype('float32') / 255
    # 以numpy物件将图像转换后的ndarray物件保存在档案系统中
    np.save(path + "/" + datasetType + suffix + ".npy", imgData)

#标签转化

def label2int(ch):
    asciiVal = ord(ch)
    if (asciiVal <= 57):  # 0-9
        asciiVal -= 48
    elif (asciiVal <= 90):  # AZ
        asciiVal -= 55
    else:  # az
        asciiVal -= 61
    return asciiVal


def int2label(i):
    if (i <= 9):  # 0-9
        i += 48
    elif (i <= 35):  # AZ
        i += 55
    else:  # az
        i += 61
    return chr(i)
y_train = pd.read_csv(path + "/trainLabels.csv").values[:, 1] #ID不取，只取标签
Y_train = np.zeros((y_train.shape[0], 62)) # AZ, az, 0-9共有62个类别
for i in range(y_train.shape[0]):
    Y_train[i][label2int(y_train[i])] = 1 #one-hot
np.save(path + "/" + "labelsPreproc.npy", Y_train)


