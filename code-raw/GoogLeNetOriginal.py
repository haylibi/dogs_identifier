import tensorflow_datasets as tfds

(train_dataset, test_dataset), info = tfds.load(
    'cats_vs_dogs',
    split = ('train[:80%]', 'train[80%:]'),
    with_info = True,
    as_supervised = True
)

len(train_dataset), len(test_dataset)

for X, y in train_dataset:
  print(X.shape, y.numpy())
  image_1 = X.numpy()
  break

import matplotlib.pyplot as plt

plt.imshow(image_1)

import tensorflow as tf

def normalize_img(image, label):
  return (tf.cast(image, tf.float32) / 255.0, label)

def resize(image, label):
  return (tf.image.resize(image, (224, 224)), label)

train_dataset = train_dataset.map(resize, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

# Pass small value to shuffle and batch if running on colab
SHUFFLE_VAL = len(train_dataset) // 1000
BATCH_SIZE = 4

train_dataset = train_dataset.shuffle(SHUFFLE_VAL)
train_dataset = train_dataset.batch(BATCH_SIZE)

train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.map(resize, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE)

test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

test_dataset

for (img, label) in train_dataset:
  print(img.numpy().shape, label.numpy())
  break

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *


def inception_module(x, base_channels=32):
  a = Conv2D(base_channels*2, 1, 1, activation='relu')(x)

  b_1 = Conv2D(base_channels*4, 1, 1, activation='relu')(x)
  b_2 = Conv2D(base_channels*4, 3, 1, padding='same', activation='relu')(b_1)

  c_1 = Conv2D(base_channels, 1, 1, activation='relu')(x)
  c_2 = Conv2D(base_channels, 5, 1, padding='same', activation='relu')(c_1)

  d_1 = MaxPooling2D(3, 1, padding='same')(x)
  d_2 = Conv2D(base_channels, 1, 1, activation='relu')(d_1)

  return Concatenate(axis=-1)([a, b_2, c_2, d_2])

inp = Input((224, 224, 3))

maps = inception_module(inp)

gap = GlobalAveragePooling2D()(maps)

output = Dense(1, activation='sigmoid')(gap)

model = Model(inputs=inp, outputs=output)

model.summary()


for (img, label) in train_dataset:
  print(model(img).numpy().shape, label.numpy())
  break

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# Learning rate is also a hyperparameter
model.compile(loss=BinaryCrossentropy(), 
              optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(patience=5, monitor='loss')

model.fit(
    train_dataset,
    epochs=100,
    validation_data=test_dataset,
    callbacks=[es]
)