import platform
import tensorflow
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from IPython.display import  Image
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPool2D
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import chars_image_util
import matplotlib.pyplot as plt
% matplotlib inline
# ImageDataGenerator构建函数需要几个参数来定义我们想要使用的增强效果。我只会通过对我们的案例有用的参数进行设定，如果您需要对您的图像进行其他修改，请参阅Keras文档。
#
# featurewise_center，featurewise_std_normalization和zca_whitening不使用，因为在本案例里它们不会增加网络的性能。如果你想测试这些选项，一定要合适地计算相关的数量，并将这些修改应用到你的测试集中进行标准化。
# rotation_range 20左右的值效果最好。
# width_shift_range 0.15左右的值效果最好。
# height_shift_range 0.15左右的值效果最好。
# shear_range 0.4 左右的值效果最好。
# zoom_range 0.3 左右的值效果最好。
# channel_shift_range 0.1左右的值效果最好。

# 模型学习(Learning)
# 对于模型的训练，我使用了分类交叉熵(cross-entropy)作为损失函数(loss function)，最后一层使用softmax的激励函数。
#
# 演算法(Algorithm)
# 在这个模型里我选择使用AdaMax和AdaDelta来作为优化器(optimizer)，而不是使用经典的随机梯度下降（SGD）算法。同时我发现AdaMax比AdaDelta在这个问题上会给出更好的结果。
#
# 但是，对于具有众多滤波器和大型完全连接层的复杂网络，AdaMax在训练循环不太收敛，甚至无法完全收敛。因此在这次的网络训练过程我拆成二个阶段。第一个阶段，我先使用AdaDelta进行了20个循环的前期训练为的是要比较快速的帮忙卷积网络的模型收敛。第二个阶段，则利用AdaMax来进行更多训练循环与更细微的修正来得到更好的模型。
#
# 如果将网络的大小除以2，则不需要使用该策略。
#
# 训练批次量(Batch Size)
# 在保持训练循环次数不变的同时，我试图改变每次训练循环的批量大小(batch size)。大的批量(batch)会使算法运行速度更快，但结果效能不佳。这可能是因为在相同数量的数据量下，更大的批量意味着更少的模型权重的更新。无论如何，在这个范例中最好的结果是在批量(batch size) 设成128的情况下达到的。
#
# 网络层的权重初始(Layer Initialization)
# 如果网络未正确初始化，则优化算法可能无法找到最佳值。我发现使用he_normal来进行初始化会使模型的学习变得更容易。在Keras中，你只需要为每一层使用kernel_initializer='he_normal'参数。
#
# 学习率衰减(Learning Rate Decay)
# 在训练期间逐渐降低学习率(learning rate)通常是一个好主意。它允许算法微调参数，并接近局部最小值。但是，我发现使用AdaMax的optimizer，在没有设定学习速率衰减的情况下结果更好，所以我们现在不必担心。
#
# 训练循环(Number of Epochs)
# 使用128的批量大小，没有学习速度衰减，我测试了200到500个训练循环。即使运行到第500个训练循环，整个网络模型似乎也没出现过拟合(overfitting)的情形。我想这肯定要归功于Dropout的设定发挥了功效。我发现500个训练循环的结果比300个训练循环略好。最后的模型我用了500个训练循环，但是如果你在CPU上运行，300个训练循环应该就足够了。
#
# 交叉验证(Cross-Validation)
#为了评估不同模型的质量和超参数的影响，我使用了蒙特卡洛交叉验证：我随机分配了初始数据1/4进行验证，并将3/4进行学习。我还使用分裂技术，确保在我们的例子中，每个类别约有1/4图像出现在测试集中。这导致更稳定的验证分数。

batch_size = 128
nb_classes = 62
nb_epoch = 300

img_height, img_width = 32, 32
path = '../data/chars74/'
X_train_all = np.load(path+"trainPreproc.npy")
Y_train_all = np.load(path+"labelsPreproc.npy")

#将资料区分为训练资料集与验证资料集
X_train,  X_val,  Y_train,  Y_val =  train_test_split ( X_train_all ,  Y_train_all ,  test_size = 0.25, stratify = np.argmax ( Y_train_all, axis = 1))
datagen = ImageDataGenerator(rotation_range=20,width_shift_range=0.15, height_shift_range=0.15, shear_range=0.4, zoom_range=0.3,channel_shift_range=0.1)
model = Sequential()

#第一层需要shape,后面的网络可以不用，因为keras会自动根据前面的网络计算
model.add(Convolution2D(128, (3, 3), padding='same', kernel_initializer='he_normal',activation='relu',input_shape=(img_height, img_width, 1)))
model.add(Convolution2D(128, (3, 3), padding='same', kernel_initializer='he_normal',activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(256, (3, 3), padding='same', kernel_initializer='he_normal',activation='relu'))
model.add(Convolution2D(256, (3, 3), padding='same', kernel_initializer='he_normal',activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(512, (3, 3), padding='same', kernel_initializer='he_normal',activation='relu'))
model.add(Convolution2D(512, (3, 3), padding='same', kernel_initializer='he_normal',activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(4096, kernel_initializer='he_normal', activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(4096, kernel_initializer='he_normal', activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes, kernel_initializer='he_normal', activation='softmax'))
print(model.summary())

#首先使用AdaDelta来做第一阶段的训练,因为AdaMax会无卡住
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=batch_size, epochs=20,validation_data=(X_val, Y_val),verbose=1)

#接着改用AdaMax
model.compile(loss='categorical_crossentropy',
              optimizer='adamax',
              metrics=["accuracy"])

#我们想要保存在训练过程中验证结果比较好的模型
saveBestModel = ModelCheckpoint ( "best.kerasModelWeights" ,  monitor = 'val_acc' ,  verbose = 1 ,  save_best_only = True ,  save_weights_only = True )
#在训练的过程透过ImageDataGenerator来持续产生图像资料
history  =  model . fit_generator ( datagen . flow ( X_train ,  Y_train ,  batch_size = batch_size ),
                    steps_per_epoch = len ( X_train ) / batch_size ,
                    epochs = nb_epoch ,
                    validation_data = ( X_val ,  Y_val ),
                    callbacks = [ saveBestModel ],
                    verbose = 1)


#预测
model.load_weights("best.kerasModelWeights")
X_test = np.load(path+'testPreproc.npy')

Y_test_pred = model.predict_classes(X_test)
#从类别的数字转换为字符
vInt2label  =  np . vectorize ( chars_image_util.int2label )
Y_test_pred  =  vInt2label ( Y_test_pred )

#保存预测结果到档案系统
np . savetxt ( path + "/jular_pred"  +  ".csv" ,  np . c_ [ range ( 6284 , len ( Y_test_pred ) + 6284 ), Y_test_pred ],  delimiter = ',' ,  header  =  'ID,Class' ,  comments  =  '' ,  fmt = ' %s ' )

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))
plt.plot(epochs, acc, 'bo',label = 'Train_acc')
plt.plot(epochs, val_acc, 'bo',label = 'Validation_acc')
plt.legend()

plt.figure()
#把"训练损失(Training loss)"与"验证损失(Validation loss)"的趋势线形表现在图表上
plt.plot( epochs ,  loss ,  'bo' ,  label = 'Training loss' )
plt.plot( epochs ,  val_loss ,  'b' ,  label = 'Validation loss' )
plt.title( 'Training and validation loss' )
plt.legend()

plt.show()