"""Model architecture definition file
"""
from keras.models import Sequential
import keras.layers as kl
from keras.regularizers import l2
from sklearn.linear_model import SGDClassifier


def logistic_regressionn_keras(units, activation='sigmoid', regularizer=l2(1.0), learning_rate=0.015):
    model = Sequential(kl.Dense(units, activation=activation,kernel_regularizer=regularizer))
    return model

def logistic_regression_scikit(loss='log', penalty='l2', alpha=1.0,
                               learning_rate='optimal'):
    clf = SGDClassifier(loss=loss, penalty=penalty, alpha=alpha,
        learning_rate=learning_rate)
    return clf


def fc_nn():
    pass
