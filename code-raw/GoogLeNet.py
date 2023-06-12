import cv2
import os
import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import Model, load_model
from keras.initializers import glorot_uniform
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import keras.backend as K

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import Concatenate, GlobalAveragePooling2D


os.chdir(r'C:\Users\Duarte\Documents\MEGA\03. Vida Académica\03. Mestrado Ciencias Computadores\1 Ano\Semestre 1\Topicos Avancados Inteligencia Artificial\Submissoes\Trabalhos\Projeto')

ROWS = 224
COLS = 224
CHANNELS = 3
CLASSES = 2

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

def prepare_data(images):
    m = len(images)
    X = np.zeros((m, ROWS, COLS, CHANNELS), dtype=np.uint8)
    y = np.zeros((1, m), dtype=np.uint8)
    for i, image_file in enumerate(images):
        X[i,:] = read_image(image_file)
        if 'dog' in image_file.lower():
            y[0, i] = 1
        elif 'cat' in image_file.lower():
            y[0, i] = 0
    return X, y

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

TRAIN_DIR = 'data/dataset_dogs_vs_cats/train/'
TEST_DIR = 'data/dataset_dogs_vs_cats/test/'

train_images = [f'{TRAIN_DIR}{i}/{j}' for i in os.listdir(TRAIN_DIR) for j in os.listdir(TRAIN_DIR+i)]
test_images =  [f'{TEST_DIR}{i}/{j}' for i in os.listdir(TEST_DIR) for j in os.listdir(TEST_DIR+i)]

train_set_x, train_set_y = prepare_data(train_images)
test_set_x, test_set_y = prepare_data(test_images)

X_train = train_set_x/255
X_test = test_set_x/255

Y_train = convert_to_one_hot(train_set_y, CLASSES).T
Y_test = convert_to_one_hot(test_set_y, CLASSES).T

print ("number of training examples =", X_train.shape[0])
print ("number of test examples =", X_test.shape[0])
print ("X_train shape:", X_train.shape)
print ("Y_train shape:", Y_train.shape)
print ("X_test shape:", X_test.shape)
print ("Y_test shape:", Y_test.shape)

def inception_module(x, base_channels=32):
    a = Conv2D(base_channels*2, 1, 1, activation='relu')(x)

    b_1 = Conv2D(base_channels*4, 1, 1, activation='relu')(x)
    b_2 = Conv2D(base_channels*4, 3, 1, padding='same', activation='relu')(b_1)

    c_1 = Conv2D(base_channels, 1, 1, activation='relu')(x)
    c_2 = Conv2D(base_channels, 5, 1, padding='same', activation='relu')(c_1)

    d_1 = MaxPooling2D(3, 1, padding='same')(x)
    d_2 = Conv2D(base_channels, 1, 1, activation='relu')(d_1)

    return Concatenate(axis=-1)([a, b_2, c_2, d_2])

inp = Input((ROWS, COLS, 3))

maps = inception_module(inp)

gap = GlobalAveragePooling2D()(maps)

output = Dense(1, activation='sigmoid')(gap)

model = Model(inputs=inp, outputs=output)

model.summary()


# Learning rate is also a hyperparameter
model.compile(loss=BinaryCrossentropy(), 
              optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])


es = EarlyStopping(patience=5, monitor='loss')

model.fit(
    X_train,
    epochs=100,
    validation_data=tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(X_test)),
    callbacks=[es]
)