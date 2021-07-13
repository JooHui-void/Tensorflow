import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])


# compile method -> (optimizer 최적화함수, loss function 손실함수 , metris 모델 성능 지표함수)

model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1, 0, 1, 2, 3, 4], dtype=float)
ys = np.array([-3, -1, 1, 3, 5, 7], dtype=float)

# training 하기
model.fit(xs, ys, epochs=500)


print(model.predict([10.0]))
# 18.98... 유사한 값 출력