"""Model architecture definition file
"""
from keras.models import Sequential
import keras.layers as kl
from keras.regularizers import l2
from keras.optimizers import Adam
import gin
from sklearn.linear_model import SGDClassifier

@gin.configurable
def logistic_regression_keras(n_features, l2_regularization=1.0, learning_rate=0.015):
    model = Sequential([
        kl.Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_regularization), input_shape=(n_features, ))
        ]
        )
    model.compile(Adam(lr=learning_rate), "binary_crossentropy", ['acc'])
    return model

def logistic_regression_scikit(loss='log', penalty='l2', alpha=1.0,
                               learning_rate='optimal'):
    clf = SGDClassifier(loss=loss, penalty=penalty, alpha=alpha,
        learning_rate=learning_rate)
    return clf

def fc_nn():
    pass
