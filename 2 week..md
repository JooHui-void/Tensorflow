# 2 week  
## Using Fashion MNIST
***
### import library

~~~
import tensorflow as tf
import numpy as np
from tensorflow import keras
~~~
### import fasion mnist
fashion_mnist는 기본적으로 텐서플로우에 있음
~~~
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels),(test_images,test_labels) = fashion_mnist.load_data() # 4개의 numpy 배열
~~~

labels은 0부터 9까지의 수이고 각각 아래의 아이템을 나타낸다
~~~
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
~~~

아래 코드로 데이터 크기 테스트가 가능하다
~~~
#len(train_labels)  #60000저장되어있음
#train_labels   #array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)  각 레이블 0~9 정수
#test_images.shape   # (10000, 28, 28)  10000개의 이미지 28 x 28 pixels
~~~

### making model
- Flatten : 이미지 2차원 -> 1차원으로 변경(변환만 함, 학습X)
- Dense : 128개의 뉴런 가짐
- 10개의 노드 (10개의 확률 반환)

~~~
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
~~~
### model compile
- optimizer은 1 week의 'sgd' 와 다른 'adam' 을 사용
     
~~~
model.compile(optimizer='adam', #데이터와 loss 바탕으로 모델 업데이트 방법 결정
              loss='sparse_categorical_crossentropy',# 모델의 오차 결정 
              metrics=['accuracy']) # 이미지 추정 비율 
~~~

### training
~~~
model.fit(train_images, train_labels,epochs=5) # 0.81의 정확도 나옴
~~~
model.summary() 를 이용하면 각 epochs 구조를 보여준다

### Identify the performance of model with a test set
~~~
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
~~~

### Predicting
~~~
print('\n테스트 정확도:', test_acc)  # 0.7983 나온다
~~~