import argparse
import datetime
import logging
import os
import pickle

import pandas as pd
from tqdm import trange
import train_with_keras
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib


def load_batch(directory, batch_size, sep=','):
    sub_batch = None
    # For keras, my generator must be able to loop infinitely
    while True:
        for file in os.listdir(directory):
            filename = directory + str(os.fsdecode(file))
            rows_df = pd.DataFrame.sample(
                pd.read_csv(filename, sep=sep, index_col=0), frac=1)

            rows_df.y = [0 if r == -1 else r for r in rows_df.y]
            sub = (rows_df.shape[0] // batch_size) + 1
            for i in range(sub):
                start = (i) * batch_size
                if sub_batch is None:
                    end = min(rows_df.shape[0], start + batch_size)
                    sub_batch = rows_df.iloc[start:end, :]
                else:
                    end = batch_size - sub_batch.shape[0]
                    sub_batch = sub_batch.append(rows_df.iloc[start:end, :])
                if sub_batch.shape[0] == batch_size:
                    yield (sub_batch.iloc[:, 1:], sub_batch.iloc[:, 0])
                    sub_batch = None


def train(arg_batch_size, arg_num_epochs, training_batches_folder,
    steps_training):

    training_generator = load_batch(training_batches_folder, arg_batch_size)
    clf = SGDClassifier(loss='log', penalty='l2', alpha=1.0,
        learning_rate='optimal')

    for e in range(arg_num_epochs):
        with trange(steps_training) as t:
            for _ in t:
                t.set_description('Iteration %i' % (e+1))
                X, y = next(training_generator)
                clf.partial_fit(X, y, classes=[0,1])

    return(clf)


def test(clf, arg_batch_size, testing_batches_folder, steps_testing):
    testing_generator = load_batch(testing_batches_folder,
        arg_batch_size*steps_testing)
    print("Evaluating", arg_batch_size*steps_testing, "samples.")
    X_test, y_test = next(testing_generator)
    score = clf.score(X_test, y_test)
    return(score)


def save(clf, clf_file):
    joblib.dump(clf, clf_file)