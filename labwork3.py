# -*- coding: utf-8 -*-
"""labwork3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AWOu6tMpa9o3KepQy94GkduMHbv2Ga14
"""

!pip3 install numba
!wget "file:///Users/khoahung/Desktop/example-orig.jpg"

import sys
import numba
from numba import cuda
import numpy as np
from numba import vectorize
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy import asarray

print("Python version:", sys.version)
print("Numba version:", numba.__version__)
print("Numpy version:", np.__version__)
print("Numpy version:",cuda.gpus.__module__)


#img = mpimg.imread('example-orig.jpg')
#imgplot = plt.imshow(img)
#plt.show()

#x_data = asarray(mpimg.imread('example-orig.jpg'))
#flatSrc = x_data.flatten()

img = cv.imread("example-orig.jpg")

im_resized = cv.resize(img, (224, 224), interpolation=cv.INTER_LINEAR)
plt.imshow(cv.cvtColor(im_resized, cv.COLOR_BGR2RGB))
plt.show()

x_data = np.array(img)
flatSrc = x_data.flatten()

imageWidth,imageHeight,c = x_data.shape
print(x_data.shape)

@cuda.jit
def grayscale(src, dst):
  tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
  g = np.uint8((src[tidx, 0] + src[tidx, 1] + src[tidx, 2]) / 3)
  dst[tidx, 0] = dst[tidx, 1] = dst[tidx, 2] = g

pixelCount = imageWidth * imageHeight
blockSize = 64
gridSize = int(pixelCount / blockSize)

devSrc = cuda.to_device(flatSrc)
devDst = cuda.device_array((pixelCount, 3), np.uint8)
grayscale[gridSize, blockSize](devSrc, devDst)
hostDst = devDst.copy_to_host()

print(flatSrc.shape)