# deep-learning-practice

학부 딥러닝 수업(23-2학기)에서 진행한 머신러닝 및 딥러닝 실습을 정리한 레포지토리.

기본적인 머신러닝 알고리즘부터 DNN, CNN, RNN까지 직접 학습하고, 모델별 성능 비교 및 다양한 학습 조건에 따른 실험을 진행했다.

## 디렉토리 구조

```
deep-learning-practice/
├── README.md
├── requirements.txt
├── .gitignore
│
├── 01-machine-learning/
│   ├── iris/
│   │   └── iris_classification.ipynb
│   ├── digits/
│   │   └── digits_classification.ipynb
│   └── insurance/
│       ├── insurance_regression.ipynb
│       └── insurance.csv
│
├── 02-dnn/
│   └── mnist-fashion-mnist/dnn_experiments.ipynb
│
├── 03-cnn/
│   └── mnist-cifar10/cnn_cifar10.ipynb, cnn_mnist.ipynb
│
└── 04-rnn/
    └── time-series/rnn_sine_wave.ipynb, simple_rnn_sequence.py
```

각 노트북 안에 목적, 데이터셋, 실험 내용, 결과, 배운 점을 마크다운으로 정리해뒀다.
파일별 자세한 설명은 노트북을 직접 열어서 확인하면 된다.

## 01. Machine Learning

- **Iris 분류**: KNN, GaussianNB, DecisionTree, AdaBoost를 같은 데이터로 비교하고,
  KNN의 `n_neighbors` 값에 따른 train/test accuracy 변화(과적합 여부)를 확인했다.
- **Digits 분류**: 8x8 손글씨 숫자(64차원 벡터)에 대해 위 네 가지 모델을 다시 비교하고,
  Confusion Matrix와 클래스별 F1-score로 정량 평가했다.
- **보험료 예측**: 성별로 데이터를 나눠 age/bmi 각각과 charges(보험료) 사이의 관계를
  선형 회귀로 살펴봤다.

## 02. DNN

MNIST 숫자 이미지를 Dense 레이어 기반 DNN(Flatten → Dense → Dense → Dense)으로 분류하면서,
학습에 영향을 주는 세 가지 조건을 각각 80 epoch씩 비교했다.

- 가중치 초기화: Random Normal vs Xavier(Glorot) Normal
- 입력 정규화: 0~255 원본 픽셀값 vs 0~1 정규화값
- Dropout 적용 여부

## 03. CNN

Conv2D + MaxPooling2D 기반 CNN을 MNIST와 CIFAR-10 두 데이터셋에 적용했다.
CIFAR-10 실험에서는 같은 모델 구조에 optimizer만 Adam / Adadelta로 바꿔가며
학습 결과를 비교했고, 예측에 성공/실패한 샘플을 이미지로 직접 확인했다.

## 04. RNN

노이즈가 섞인 사인파(sine wave) 시계열 데이터를 SimpleRNN으로 학습시켜 다음 값을 예측했다.
슬라이딩 윈도우 방식으로 시계열을 지도학습 형태(과거 값 → 다음 값)로 바꾸는 전처리 과정을 다뤘다.

## 기술 스택

| 구분 | 사용 기술 |
|---|---|
| Language | Python |
| Machine Learning | scikit-learn |
| Deep Learning | TensorFlow, Keras |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib |
| Experiment | TensorBoard |
| Environment | Jupyter Notebook |

## 학습 내용

- 머신러닝 모델의 학습 과정과 성능 평가 방법 이해
- 데이터 정규화, 가중치 초기화, Dropout 등 학습 조건에 따른 결과 비교
- DNN, CNN, RNN의 기본 구조와 데이터 특성에 따른 활용 방식 학습
- Accuracy, Confusion Matrix, F1-score 등을 활용한 모델 성능 평가 경험
- 이미지 및 시계열 데이터의 특성에 맞는 전처리와 학습 과정 경험
- 다양한 실험 조건을 설정하고 결과를 시각화하여 비교·분석하는 경험
