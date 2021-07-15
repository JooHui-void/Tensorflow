import cv2
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

i = misc.ascent()
plt.subplot(1,4,1)

plt.gray()
plt.axis('off')
plt.imshow(i)
i_transformed = np.copy(i)
size_x = i_transformed.shape[0]
size_y = i_transformed.shape[1]

filter = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
          [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
          [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]

weight = 1

for k in range(3):
    plt.subplot(1, 4, k+2)
    i_transformed = np.copy(i)
    for x in range(1,size_x-1):
        for y in range(1,size_y-1):
            convolution =0.0
            convolution = convolution + (i[x - 1, y-1] * filter[k][0][0])
            convolution = convolution + (i[x, y-1] * filter[k][0][1])
            convolution = convolution + (i[x + 1, y-1] * filter[k][0][2])
            convolution = convolution + (i[x-1, y] * filter[k][1][0])
            convolution = convolution + (i[x, y] * filter[k][1][1])
            convolution = convolution + (i[x+1, y] * filter[k][1][2])
            convolution = convolution + (i[x-1, y+1] * filter[k][2][0])
            convolution = convolution + (i[x, y+1] * filter[k][2][1])
            convolution = convolution + (i[x+1, y+1] * filter[k][2][2])
            convolution = convolution * weight
            if(convolution<0):
                convolution = 0
            if(convolution>255):
                couvolution = 255
            i_transformed[x,y] = convolution

    plt.imshow(i_transformed)

plt.show()