import math

import inline as inline
import matplotlib as matplotlib
import numpy as np
import matplotlib.pyplot as plt
import h5py


m_train = 1
m_test = 1

def base_sigmoid(x):
    s = 1.0 / (1.0 + math.exp(-1 * x))
    return s;


def base_tanh(x):
    s = math.tanh(x)
    return s;


def Sigmoid(matris):
    result = 1 / (1 + np.exp((-1) * matris))
    return result;


def Tanh(matris):
    result = np.tanh(matris)
    return result;


def Sigmoid_Derivative(matris):
    date = Sigmoid(matris)
    return date * (1 - date);


def Tanh_Derivative(matris):
    date = Tanh(matris)
    return (1 - date * date);


def ConvertorImageToVector(image):
    vector = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)
    return vector;


def NormalizeRows(matris):
    date = np.linalg.norm(matris, axis=1, keepdims=True)
    return matris / date;

def L1(yhat , y):
    t = np.abs(yhat , y)
    return np.sum(t);

def L2(yhat , y):
    t = np.abs(yhat , y)
    return np.sum(np.multiply(t,t));

def Initializing_Parameters():

    ##train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes;
    ##m_test = len(test_set_y)
    ##m_train = len(train_set_y)
    return;

def Initialize_With_Zeros(dim):

    w = np.zeros((dim , 1))
    b = 0
    assert(w.shape == (dim , 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w , b;

def Propagate(w, b, X, Y):

    m = X.shape[1]
    A = Sigmoid(np.dot(np.transpose(w),X) + b)                                    # compute activation
    temp = np.multiply(Y , np.log(A)) + np.multiply(1 - Y,np.log(1 - A))
    cost = (-1/m) * np.sum(temp)                               # compute cost
    dz = A - Y
    dw = (1/m) * np.dot(X,np.transpose(dz))
    db = (1/m) *np.sum(dz)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):

    for i in range(num_iterations):
        grads, cost = Propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads;

def predict(w, b, X):

    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)

    A = Sigmoid(np.dot(np.transpose(w),X) + b)

    for i in range(A.shape[1]):

        if A[0 ,i] >= 0.5 :
            Y_prediction[0 ,i] = 1
        if A[0,i] < 0.5:
            Y_prediction[0 , i] = 0

    assert(Y_prediction.shape == (1, m))

    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):

    w, b = np.zeros(X_train.shape) , 0

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost = False)

    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train =  predict(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))


    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train" : Y_prediction_train,
         "w" : w,
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}

    return d
