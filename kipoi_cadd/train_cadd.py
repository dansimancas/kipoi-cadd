import dask
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
from dask_ml.linear_model import LogisticRegression
import numpy as np
import argparse
import pickle


def confusion_matrix_dask(truth, predictions, labels_list=[]):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    if not labels_list:
        TP = (truth[predictions == 1] == 1).sum()
        FN = (truth[predictions != 1] == 1).sum()
        TN = (truth[predictions != 1] != 1).sum()
        FP = (truth[predictions == 1] != 1).sum()
    for label in labels_list:
        TP = (truth[predictions == label] == label).sum() + TP
        FN = (truth[predictions != label] == label).sum() + FP
        TN = (truth[predictions != label] != label).sum() + TN
        FP = (truth[predictions == label] != label).sum() + FN

    TN, FP, FN, TP = dask.compute(TN, FP, FN, TP)
    return(TN, FP, FN, TP)


def split(full_set):
    training_dd = dd.read_csv(full_set, assume_missing=True)

    y = training_dd.y
    X = training_dd.drop('y', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, test_size=0.2)
    return(X_train, X_test, y_train, y_test)


def train(X_train, y_train, out_model):
    lr = LogisticRegression(
        penalty='l2', solver='lbfgs', n_jobs=64, max_iter=10)
    # If leave just the dataframe, will throw an error saying "This estimator
    # does not support dask dataframes."
    lr.fit(X_train.values, y_train.values)

    # Saving model for later prediction
    pickle.dump(lr, open(out_model, "wb"))

    # Outputing some statistics
    y_train_pred = lr.predict(X_train.values)

    TN, FP, FN, TP = confusion_matrix_dask(y_train.values, y_train_pred)
    print("Read like \n[[TN, FP], \n[FN, TP]]\n",
          np.array([[TN, FP], [FN, TP]]))


def test(X_test, y_test, model):
    # lr.predict returns a Dask array so don't have to do .values to use
    # confusion_matrix_dask
    lr = pickle.load(open(model, "rb"))
    y_test_pred = lr.predict(X_test.values)

    TN, FP, FN, TP = confusion_matrix_dask(y_test.values, y_test_pred)
    print("Read like \n[[TN, FP], \n[FN, TP]]\n")
    print(np.array([[TN, FP], [FN, TP]]))
    sum([TP, FP, TN, FN])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train a linear regression model for CADD using Dask.")
    parser.add_argument('train_set', help="Input training file.")
    parser.add_argument('out_model', help="Full path of out_model trained \
    model.")
    parser.add_argument('--params', help="Parameters to initialize the \
    LogisticRegression model.")
    parser.add_argument('--log', help="Verbosity of logs.")
    args = parser.parse_args()

    if not (args.train_set and args.out_model):
        args.train_set = ("/s/project/kipoi-cadd/data/raw/v1.3/" +
                          "training_data/training_data.imputed.csv")
        args.out_model = ("/s/project/kipoi-cadd/data/processed/" +
                          "kipoi_cadd_models/lr.pickle")

    X_train, X_test, y_train, y_test = split(args.train_set)
    train(X_train, y_train, args.out_model)
    test(X_test, y_test, args.out_model)
