from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten,Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import random

# 필수 모듈들을 import 합니다.

def load_data():
    # 먼저 MNIST 데이터셋을 로드하겠습니다.
    # 케라스는 `keras.datasets`에 널리 사용하는 데이터셋을 로드하기 위한 함수를 제공합니다.
    # 이 데이터셋은 이미 훈련 세트와 테스트 세트로 나누어져 있습니다.
    # 훈련 세트를 더 나누어 검증 세트를 만드는 것이 좋습니다:

    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    X_train_full = X_train_full.astype(np.float32)
    X_test = X_test.astype(np.float32)
    #print(X_train_full.shape, y_train_full.shape)
    #print(X_test.shape, y_test.shape)
    return X_train_full, y_train_full, X_test, y_test

def data_normalization(X_train_full, X_test):
    # 전체 훈련 세트를 검증 세트와 (조금 더 작은) 훈련 세트로 나누어 보죠. 또한 픽셀 강도를 255로 나누어 0~1 범위의 실수로 바꾸겠습니다.

    X_train_full = X_train_full / 255.

    X_test = X_test / 255.
    train_feature = np.expand_dims(X_train_full, axis=3)
    test_feature = np.expand_dims(X_test, axis=3)
    # Tensorflow를 사용하기 위해 hannel 값을 추가 해줍니다.

    print(train_feature.shape, train_feature.shape)
    print(test_feature.shape, test_feature.shape)

    return train_feature,  test_feature


def draw_digit(num):
    for i in num:
        for j in i:
            if j == 0:
                print('0', end='')
            else :
                print('1', end='')
        print()

def makemodel(X_train, y_train, X_valid, y_valid, weight_init):
    model = Sequential()
    # Sequential 모델을 생성합니다.
    model.add(Conv2D(32, kernel_size=(3, 3),  input_shape=(28,28,1), activation='relu'))
    # 필터의 크기 = 3, 활성화 함수는 relu, 입력 모양은 28 * 28인 Conv2D층을 추가합니다.
    model.add(MaxPooling2D(pool_size=2))
    # MaxPooling 층을 추가하고, pooling filter의 size는 2로 지정합니다.
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    # 위의 작업을 반복합니다.
    model.add(Dropout(0.25))
    # 성능 향상을 위해 Dropout 해줍니다. (Dropout = 선택적으로 노드를 Drop시켜, 학습의 과적합을 막는다.)
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    # Flatten 후, 각각 relu, softmax 활성 함수를 사용하는 층을 2개 추가 해줍니다.
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

    


def plot_history(histories, key='accuracy'):
    plt.figure(figsize=(16,10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                       '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()

    plt.xlim([0,max(history.epoch)])
    plt.show()
    # epoch의 진행과정을 나타내주는 함수입니다.

def draw_prediction(pred, k,X_test,y_test,yhat):
    samples = random.choices(population=pred, k=16)

    count = 0
    nrows = ncols = 4
    plt.figure(figsize=(12,8))

    for n in samples:
        count += 1
        plt.subplot(nrows, ncols, count)
        plt.imshow(X_test[n].reshape(28, 28), cmap='Greys', interpolation='nearest')
        tmp = "Label:" + str(y_test[n]) + ", Prediction:" + str(yhat[n])
        plt.title(tmp)

    plt.tight_layout()
    plt.show()
    # 예측된 결과를 그려주는 함수입니다. 

def evalmodel(X_test,y_test,model):
    yhat = model.predict(X_test)
    yhat = yhat.argmax(axis=1)
    # x_test에 대한 예측값을 나타내는 yhat을 생성합니다.

    print(yhat.shape)
    answer_list = []

    for n in range(0, len(y_test)):
        if yhat[n] == y_test[n]:
            answer_list.append(n)

    draw_prediction(answer_list, 16,X_test,y_test,yhat)

    answer_list = []

    for n in range(0, len(y_test)):
        if yhat[n] != y_test[n]:
            answer_list.append(n)

    draw_prediction(answer_list, 16,X_test,y_test,yhat)
    # x_test와 yhat을 비교해줍니다.

def main():
    X_train, y_train, X_test, y_test = load_data()
    # 모듈을 통해 불러온 MNIST 데이터를 train 부분과 test 부분으로 분류합니다.
    # x = 28 * 28의 data 값, y = x에 해당하는 label 값
    
    X_train, X_test = data_normalization(X_train,  X_test)
    # 0 ~ 255 사이인 x 값들을 0 ~ 1 사이로 정규화 시켜줍니다.

    model= makemodel(X_train, y_train, X_test, y_test,'glorot_uniform')
    # glorot_uniform 형식으로 가중치를 초기화하여 model을 만들어냅니다.

    baseline_history = model.fit(X_train,
                                 y_train,
                                 epochs=50,
                                 batch_size=512,
                                 validation_data=(X_test, y_test),
                                 verbose=2)

    evalmodel(X_test, y_test, model)
    plot_history([('baseline', baseline_history)])
    # 만들어진 model에 test 샘플들을 fit하여 실행한 후, model의 성능을 평가합니다.

main()
