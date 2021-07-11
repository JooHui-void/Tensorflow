#1 week  tensorflow 기초 :octocat: 
##Predicting values of x and y
### import library
~~~
import tensorflow as tf
import numpy as np
from tensorflow import keras
~~~
### Making model
~~~
model = tf.keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])]) 
~~~
- 1개의 레이어, 1개의 층, 1개의 뉴런
(layer은 add(), pop()을 통해 점진적 변화을 줄 수 있음)
- 빌드 후 model.summery() 함수 사용가능<br><br>
- Dense 함수란?
    - 신경망 구조의 가장 기본적인 형태 y = f(Wx + b)
        - 이때 x는 입력벡터, b는 편향벡터, W는 가중치 행렬이 된다
        - 여러 인자를 통해 가중치와 편향 초기화 가능 
          
        sdf| sdf
        --- | ---
        units | 출력값의 크기 (integer 혹은 long 형태)
        activation | 활성화 함수
        use_bias | 편향을 사용할지 여부의 boolean값
        kernel_initializer | 가중치 초기화 함수
        kernel_regularizer | 가중치 정규화 방법
        bias_regularizer | 편향 정규화 방법
        activity_regularizer | 출력값 정규화 방법
        kernel_constraint | Optimizer에 의해 업데이트 후 가중치에 적용되는 부가적인 제약함수
        bias_constraint | Optimizer에 의해 업데이트된 이후에 편향에 적용되는 부가적인 제약함수
        위의 model에서는 출력값의 크기가 1 이고 input data의 형태가 1개짜리임을 의미함

### Model compile
- Loss function -> measure model error while training
- Optimizer -> Determine how models are update
- metrics -> use for monitertin training and testing 
~~~
model.compile(optimizer ='sgd',loss ='mean_squared_error')
~~~
- optimizer에는 여러 종류가 있지만, sgd는 가장 기본적인 신경망 학습으로 확률적 경사 하강법(Stochastic Gradient Descent)이다
또다른 optimizer인 BGD와 비교하자면, BGD는 오차를 구할 때 전체 데이터를 고려하는 반면, SGD는
랜덤 선택 데이터 하나만 고려하여 계산한다.
정확도는 낮다고 판단되지만 계산이 빠르다는 장점이 있다
위의 중첩으로 미니 배치 경사 하강법이 있는데, 뒤에 batch_size= ''을 지정해주어서 사용한다
- loss는 실제값과 예측값 사이의 평균 제곱 오차인 rms를 사용했다
###Training model
~~~
model.fit(xs,ys,epochs=500)
~~~
###Predicting
이후에 x = 10일때 y값을 출력해본다면 
~~~
print(model.predict([10.0]))
~~~
18.9라는 비슷한 값이 나옴을 확인할 수 있다.

