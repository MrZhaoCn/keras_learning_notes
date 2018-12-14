import platform
import tensorflow
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from IPython.display import  Image
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Flatten,Input
from keras.layers import Conv2D, MaxPool2D
from keras.utils import plot_model
from keras.applications.vgg16 import VGG16, preprocess_input,decode_predictions
from keras.preprocessing.image import load_img, img_to_array
model = VGG16()
img_file = './evimg16tf.jpg'
image = load_img(img_file, target_size=(224, 224))
image = img_to_array(image)
print(image.shape)

#(batch_size, img_height, img_width, img_channels)
image = image.reshape((1,image.shape[0], image.shape[1], image.shape[2]))
print(image.shape)

#跟VGG16一样对图像做前置处理
image = preprocess_input(image)

y_pred = model.predict(image)
label = decode_predictions(y_pred)

#检索最高的概率
label = label[0][0]





