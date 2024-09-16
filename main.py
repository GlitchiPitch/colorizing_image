from tensorflow import keras
from tensorflow.keras.layers import (
    Input, Conv2D, UpSampling2D
)
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb

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

def proceseed_image(img):
  image = img.resize((256, 256), Image.BILINEAR)
  image = np.array(image, dtype=float)
  size = image.shape
  lab = rgb2lab(1.0/255*image)
  x, y = lab[:, :, 0], lab[:, :, 1:]

  y /= 128
  x = x.reshape(1, size[0], size[1], 1)
  y = y.reshape(1, size[0], size[1], 2)

  return x, y, size

img = Image.open("/images/image.png")
x, y, size = proceseed_image(img)