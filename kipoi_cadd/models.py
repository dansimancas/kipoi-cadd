"""Model architecture definition file
"""
from keras.models import Sequential
import keras.layers as kl


def logistic_regression(regularization):
    model = Sequential(kl.Dense(10))
    model.compile()
    return model


def fc_nn():
    pass
