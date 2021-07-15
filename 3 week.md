# 3 week  tensorflow 기초 :octocat:
## Convolution
***
### What is convolution?
  
- 필터의 값을 이용해 이미지를 변환시키는 것
- 필터의 한 부분(ex) 2x2, 3x3)을 잡아서 필터의 값 곱한 후 더해서 New pixel value를 생성
~~~
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
~~~
### Customizing this code with convolution
~~~
model = keras.Sequential([
    keras.layers.Conv2D(64, (3,3), activation='relu',
                        input_shape = (28,28,1)), # 뒤의 1 은 컬러채널인데 흑백이므로 1 하나가 들어간다
    tf.keras.layers.MaxPooling2D(2, 2), # 2 x 2 픽셀을 잡아서 최대값을 뽑아낸다. -> pixel 이 1/4이 됨
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2).
    tf.keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
~~~
- conv2D 앞에 들어있는 64, (3,3) 은 각가 filter, kernal_size 라 하는데,
filter은 convovlution의 output 개수를 의미,kernal_size는 앞의  2D convolution window의 가로 세로 길이를 의미한다. 
- 왜 output의 수가 64개인지는 아직은 알 수 없음...

### Compiling & Training
~~~
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(training_images, training_labels, epochs=5)
test_loss = model.evaluate(test_images, test_labels)
~~~

### Testing with convolution
***
~~~
f, axarr = plt.subplots(3,5)
~~~
3 x 5 크기의 subplot 생성
~~~
layer_outputs = [layer.output for layer in model.layers] # 아마 레이어 각각 output 반환..
~~~
모델 각 레이어 별 output값을 layer_outputs에 저장
~~~
activation_model = tf.keras.models.Model(inputs = model.input, outputs =layer_outputs)
for x in range(0,4): 
  f1 = activation_model.predict(test_images[0].reshape(1,28,28,1))[x] # 첫번째 테스트 이미지가 변환돼서 들어간다,
  # 리턴은 4번의 레이어 통과에 따른 output 이미지
  axarr[0,x].imshow(f1[0,:,:,1])
  f2 = activation_model.predict(test_images[3].reshape(1,28,28,1))[x]
  axarr[1,x].imshow(f2[0,:,:,1])
  f3 = activation_model.predict(test_images[6].reshape(1,28,28,1))[x]
  axarr[2,x].imshow(f3[0,:,:,1])

axarr[0,4].imshow(test_images[0])
axarr[1,4].imshow(test_images[3])
axarr[2,4].imshow(test_images[6])
plt.show()
~~~
위에서 만든 subplot 에다 과정별 layer output 이미지 저장

## Image filtering
***
### Importing
~~~
import cv2
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
~~~

~~~
i = misc.ascent()
~~~
이미지 불러옴
~~~
plt.subplot(1,4,1)
plt.gray()
plt.axis('off')
plt.imshow(i)
~~~
첫번쨰 subplot 에다가 misc.ascent()에서 가져온 이미지 넣어준다
~~~
i_transformed = np.copy(i)
size_x = i_transformed.shape[0]
size_y = i_transformed.shape[1]
~~~
- i_transformed 에 그림 복사해서 저장
- 그림의 가로 세로 저장해둠
~~~
filter = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
          [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
          [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]
~~~
필터링할때 들어갈 각종 필터 3종류 
~~~
weight = 1
~~~
이 예시에서는 weight는 상관없으므로 1로 둠
~~~
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
~~~
![image_filter](https://user-images.githubusercontent.com/69834729/125747952-aa6db279-ad88-4ae4-b4ad-b99e87ed6d76.png)