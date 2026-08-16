# SimpleRNN - 정수 시퀀스 다음 값 예측 (간단 토이 예제)
#
# rnn_sine_wave.ipynb보다 훨씬 단순한 예제. 연속된 정수 4개를 보고 그다음 정수를 예측하도록
# SimpleRNN을 학습시킨다. RNN 입출력 형태(입력 3차원, 시퀀스 길이/스텝 개념)를 가장 작은 단위로 확인해보기 위한 코드.
#
# 참고: 이 코드를 정리한 환경에는 TensorFlow가 설치되어 있지 않아 실제로 실행해서 결과를 담지는 못했다.
# 직접 실행하면 model.predict(X)와 model.predict(X_test) 결과가 각 입력의 "다음 값"에 가까운 숫자로 나오는지 확인할 수 있다.

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Activation

# 학습 데이터 생성: [0,1,2,3]->0.4, [1,2,3,4]->0.5, ... 식으로
# "연속된 4개 숫자(0.1 단위로 스케일링)를 보고 그다음 숫자를 예측" 하는 문제를 직접 만든다.
X = []
Y = []
for i in range(6):
    lst = list(range(i, i + 4))
    X.append(list(map(lambda c: [c / 10], lst)))  # 각 값을 [값] 형태로 감싸서 (timesteps, features) 구조를 만든다
    Y.append((i + 4) / 10)

X = np.array(X)   # shape: (6, 4, 1) -> 샘플 6개, 시퀀스 길이 4, 특징 1개
Y = np.array(Y)
print(X)
print(Y)

# SimpleRNN(20): 은닉 유닛 20개, 시퀀스 전체를 다 본 뒤 마지막 출력만 사용(return_sequences=False)
model = Sequential()
model.add(SimpleRNN(20, return_sequences=False, input_shape=(4, 1)))
model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=200, verbose=2)

print(model.predict(X))

# 학습에 없던 새로운 시퀀스로 예측 테스트: [0.8, 0.9, 1.0, 1.1] 다음 값을 예측 (기대값 대략 1.2 근처)
X_test = np.array([[[0.8], [0.9], [1.0], [1.1]]])
print(model.predict(X_test))
