import numpy as np
import time
from TWSVM import TWSVM_main


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    
    def y(x):
        return (scale * x) * (x > 0) + (scale * alpha * (np.exp(x) - 1)) * (x <= 0)
    
    result = y(x)
    return result

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.e ** (-x))

def hardlim(x):
    return (np.sign(x) + 1) / 2

def tribas(x):
    return np.maximum(1 - np.abs(x), 0)

def radbas(x):
    return np.exp(-(x**2))

def leaky_relu(x):
    x[x >= 0] = x[x >= 0]
    x[x < 0] = x[x < 0] / 10.0
    return x

import numpy as np


def one_hot(x, n_class):
    y = np.zeros([len(x), n_class])
    U_dataY_train = np.array([0,1])
    for i in range(n_class):
        idx= (x == U_dataY_train[i])
        y[idx,i]=1
    return y

#idx= np.array([np.where(U_dataY_train(i) == x)])


def RVFL_train(train_data, test_data, d1, d2,N,activation):
    # np.random.seed(2)
    

    # start = time.time()
    trainX=train_data[:,:-1]
    trainY=train_data[:,-1]

    s = 0.1
    
    Nsample, Nfea = trainX.shape
    U_trainY = np.unique(trainY)
    nclass = len(U_trainY)
    # dataY_train_temp = one_hot(trainY,nclass)

    W = np.random.rand(Nfea, N) * 2 - 1
    b = s * np.random.rand(1, N)
    X1 = np.dot(trainX, W) + np.tile(b, (Nsample, 1))

    if activation == 1:
        X1 = selu(X1)
    elif activation == 2:
        X1 = relu(X1)
    elif activation == 3:
        X1 = sigmoid(X1)
    elif activation == 4:
        X1 = np.sin(X1)
    elif activation == 5:
        X1 = hardlim(X1)
    elif activation == 6:
        X1 = tribas(X1)
    elif activation == 7:
        X1 = radbas(X1)
    elif activation == 8:
        X1 = np.sign(X1)
    elif activation == 9:
        X1 = leaky_relu(X1)
    elif activation == 10:
        X1 = np.tanh(X1)
    
    X = np.concatenate((trainX, X1), axis=1)

    TrainingSet = np.hstack((X, trainY.reshape(trainY.shape[0], 1)))  # Bias in the output layer

    # Test

    X=test_data[:,:-1]
    Y=test_data[:,-1]
    
    Nsample = X.shape[0]

    # Test Data

    X1 = np.dot(X, W) + np.tile(b, (Nsample, 1))

    if activation == 1:
        X1 = selu(X1)
    elif activation == 2:
        X1 = relu(X1)
    elif activation == 3:
        X1 = sigmoid(X1)
    elif activation == 4:
        X1 = np.sin(X1)
    elif activation == 5:
        X1 = hardlim(X1)
    elif activation == 6:
        X1 = tribas(X1)
    elif activation == 7:
        X1 = radbas(X1)
    elif activation == 8:
        X1 = np.sign(X1)
    elif activation == 9:
        X1 = leaky_relu(X1)
    elif activation == 10:
        X1 = np.tanh(X1)

    # X1 = np.hstack((X1, np.ones((Nsample, 1))))
    XX = np.hstack((X, X1))
    TestSet=np.hstack((XX, Y.reshape(Y.shape[0], 1)))

    EVAL_Validation, Time = TWSVM_main(TrainingSet, TestSet, d1, d2)

    # Validation_label = np.argmax(rawScore, axis=1) 

    # EVAL_Validation = Evaluate(Y, Validation_label)
    # end = time.time()
    # Time=end - start
    return EVAL_Validation,Time
