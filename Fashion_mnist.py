import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels),(test_images,test_labels) = fashion_mnist.load_data() # 4개의 numpy 배열


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# len(train_labels)  #60000저장되어있음
# train_labels   #array([9, 0, 0, ..., 3, 0, 5], dtype=unit8)  각 레이블 0~9 정수
# test_images.shape   # (10000, 28, 28)  10000개의 이미지 28 x 28 pixels

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    # Flatten : 이미지 2차원 -> 1차원으로 변경(변환만 함, 학습X)
    keras.layers.Dense(128, activation='relu'),
    # Dense : 128개의 노드
    keras.layers.Dense(10, activation='softmax')
    # 10개의 노드(10개의 확률 반환)
])

model.compile(optimizer='adam', # 데이터와 loss 바탕으로 모델 업데이트 방법 결정
              loss='sparse_categorical_crossentropy', # 모델의 오차 결정
              metrics=['accuracy']) # 이미지 추정 비율

# 훈련
model.fit(train_images, train_labels, epochs=5) # 0.81의 정확도 나옴
model.summary()
# 테스트 세트 모델의 성능 비교
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\n테스트 정확도:', test_acc)  # 0.7983 나온다

# plt.figure()
# plt.show(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()
