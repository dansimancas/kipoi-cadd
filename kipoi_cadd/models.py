"""Model architecture definition file
"""
from keras.models import Sequential
import keras.layers as kl
from keras.regularizers import l2
from keras.optimizers import Adam
import gin
from sklearn.linear_model import SGDClassifier, LogisticRegression
from keras.metrics import binary_accuracy, mae

@gin.configurable
def logistic_regression_keras(n_features, l2_regularization=1.0, learning_rate=0.015):
    model = Sequential([
        kl.Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_regularization), input_shape=(n_features, ))
        ]
        )
    model.compile(Adam(lr=learning_rate), "binary_crossentropy", [mae, binary_accuracy])
    return model

@gin.configurable
def sgd_classifier(alpha=1.0, max_iter=None, tolerance=None, verbose=True,
                               epsilon=0.1, n_jobs=1, learning_rate='optimal',
                               early_stopping=False, n_iter_no_change=5, n_iter=100):

    clf = SGDClassifier(loss='log', penalty='l2', alpha=alpha, max_iter=max_iter,
                        tol=tolerance, verbose=True, epsilon=epsilon, n_jobs=None,
                        learning_rate=learning_rate, early_stopping=early_stopping,
                        n_iter_no_change=n_iter_no_change, n_iter=n_iter)
    return clf

@gin.configurable
def logistic_regression_scikit(C=1.0, tolerance=0.0001, solver='lbfgs', max_iter=10, verbose=1,
                               warm_start=False, n_jobs=None):
    clf = LogisticRegression(penalty='l2', dual=False, tol=tolerance, C=C, random_state=1,
                             solver=solver, max_iter=max_iter, multi_class='ovr', verbose=verbose,
                             warm_start=warm_start, n_jobs=n_jobs)
    return clf

def fc_nn():
    pass
