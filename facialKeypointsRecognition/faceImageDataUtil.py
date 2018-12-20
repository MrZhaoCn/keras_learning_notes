import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Flatten,Conv2D,MaxPooling2D
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from collections import OrderedDict
from sklearn.model_selection import train_test_split

#资料路径
FTRAIN  =  '../data/facialKeypointsRecognition/training.csv'
FTEST  =  '../data/facialKeypointsRecognition/test.csv'
FLOOKUP  =  '../data/facialKeypointsRecognition/IdLookupTable.csv'

def load(test = False, cols=None):
    fname = FTEST if test else FTRAIN
    df = pd.read_csv(fname)

    # Image欄位有像素的資料(pixel values)並轉換成 numpy arrays
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    if cols:
        df = df[list(cols) + ['Image']]
    print(df.count())

    df = df.dropna()
    X = np.vstack(df['Image'].values) / 255
    X = X.astype(np.float32)

    if not test:
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48
        X, y = shuffle(X, y, random_state=42)
    else:
        y = None
    return X, y

def load2d(test = False, cols = None):
    X, y = load(test=test, cols=cols)
    X = X.reshape(-1, 96, 96, 1)
    return X, y


# 擴展keras的ImageDataGenerator來產生更多的圖像資料
class FlippedImageDataGenerator(ImageDataGenerator):
    # 由於臉部的關鍵點是左右對應的, 我們將使用鏡像(flip)的手法來產生圖像
    flip_indices = [(0, 2), (1, 3), (4, 8), (5, 9),
                    (6, 10), (7, 11), (12, 16), (13, 17),
                    (14, 18), (15, 19), (22, 24), (23, 25)]

    def next(self):
        X_batch, y_batch = super(FlippedImageDataGenerator, self).next()
        batch_size = X_batch.shape[0]
        # 隨機選擇一些圖像來進行水平鏡像(flip)
        indices = np.random.choice(batch_size, batch_size / 2, replace=False)
        X_batch[indices] = X_batch[indices, :, :, ::-1]

        # 對於有進行過水平鏡像的圖像, 也把臉部關鍵座標點進行調換
        if y_batch is not None:
            y_batch[indices, ::2] = y_batch[indices, ::2] * -1

            for a, b in self.flip_indices:
                y_batch[indices, a], y_batch[indices, b] = (
                    y_batch[indices, b], y_batch[indices, a]
                )

        return X_batch, y_batch

def cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3),
                     padding='same', activation='relu',
                     kernel_initializer='he_normal', input_shape=(96, 96, 1)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(30))  # 因為有15個關鍵座標(x,y), 共30個座標點要預測

    return model

model = cnn_model()
print(model.summary())
sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

# 載入模型訓練資料
X, y = load2d()

# 進行資料拆分
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 設定訓練參數
epochs = 100

# 產生一個圖像產生器instance
flipgen = FlippedImageDataGenerator()

# 開始訓練
history = model.fit_generator(flipgen.flow(X_train, y_train),
                               steps_per_epoch=len(X_train),
                               epochs=epochs,
                               validation_data=(X_val, y_val)
                              )

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(loss, linewidth=3, label='train')
plt.plot(val_loss, linewidth=3, label='valid')
plt.grid()
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.ylim(1e-3, 1e-2)
plt.yscale('log') # 由於整個loss的range很大, 我們使用'log'尺規來轉換
plt.show()

X_test ,_ = load2d(test=True)
y_pred = model.predict(X_test)

def plot_sample(x, y, axis):
    img = x.reshape(96,96)
    axis.imshow(img, cmap='gray')
    # 把模型預測出來的15個臉部關鍵點打印在圖像上
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)
# 打印一個6x6的圖像框格
fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
# 選出測試圖像的前16個進行視覺化
for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    plot_sample(X_test[i], y_pred[i], ax)

plt.show()

columns = ["left_eye_center_x", "left_eye_center_y", "right_eye_center_x", "right_eye_center_y",
           "left_eye_inner_corner_x", "left_eye_inner_corner_y", "left_eye_outer_corner_x", "left_eye_outer_corner_y",
           "right_eye_inner_corner_x", "right_eye_inner_corner_y", "right_eye_outer_corner_x",
           "right_eye_outer_corner_y", "left_eyebrow_inner_end_x", "left_eyebrow_inner_end_y",
           "left_eyebrow_outer_end_x", "left_eyebrow_outer_end_y", "right_eyebrow_inner_end_x",
           "right_eyebrow_inner_end_y", "right_eyebrow_outer_end_x", "right_eyebrow_outer_end_y", "nose_tip_x",
           "nose_tip_y", "mouth_left_corner_x", "mouth_left_corner_y", "mouth_right_corner_x", "mouth_right_corner_y",
           "mouth_center_top_lip_x", "mouth_center_top_lip_y", "mouth_center_bottom_lip_x", "mouth_center_bottom_lip_y"]
y_pred_final = y_pred * 48 + 48
y_pred_final = y_pred_final.clip(0, 96)  # 因為圖像為96x96, 因此座標點只能落在[0~96]
df = pd.DataFrame(y_pred_final, columns=columns)
lookup_table = pd.read_csv(FLOOKUP)
values = []

for index, row in lookup_table.iterrows():
    values.append((row['RowId'], df.iloc[row.ImageId - 1][row.FeatureName]))

now_str = datetime.now().isoformat().replace(':', '-')
submission = pd.DataFrame(values, columns=('RowId', 'Location'))
filename = 'submission-{}.csv'.format(now_str)
submission.to_csv(filename, index=False)
print("Wrote {}".format(filename))