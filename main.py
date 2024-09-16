from tensorflow import keras
from tensorflow.keras.layers import (
    Input, Conv2D, UpSampling2D
)

model = keras.Sequential([
  Input(shape=(None, None, 1)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same', strides=2),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same', strides=2),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    Conv2D(256, (3, 3), activation='relu', padding='same', strides=2),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    Conv2D(2, (3, 3), activation='tanh', padding='same'),
    UpSampling2D((2, 2))
])

model.compile(optimizer='adam', loss='mse')