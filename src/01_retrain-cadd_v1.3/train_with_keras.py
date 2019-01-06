import argparse
import datetime
import logging
import os
import pickle

import pandas as pd
import train_with_keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.regularizers import l2


def load_batch(directory, batch_size, sep=','):
    sub_batch = None
    # For keras, my generator must be able to loop infinitely
    while True:
        for file in os.listdir(directory):
            filename = directory + str(os.fsdecode(file))
            rows_df = pd.read_csv(filename, sep=sep, index_col=0)
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


def train(arg_loss, arg_metrics, arg_batch_size, arg_num_epochs,
          training_batches_folder, validation_batches_folder, steps_training,
          steps_validation):

    loss = 'binary_crossentropy' if arg_loss is None else arg_loss
    metrics = ['accuracy'] if arg_metrics is None else arg_metrics.split(',')

    # Build the model
    output_dim = 1  # One binary class
    input_dim = 1063  # number of features of the input
    model = Sequential()
    model.add(Dense(output_dim, input_dim=input_dim, activation='sigmoid',
                    kernel_regularizer=l2(1.0)))
    optimizer = SGD(lr=0.015)

    
    training_generator = load_batch(training_batches_folder, arg_batch_size)
    validation_generator = load_batch(validation_batches_folder,
                                      arg_batch_size)

    """
    *CADD v1.3 Release Notes*
    Learner: For this version we used the Logistic Regression module of
    GraphLab Create v1.4 (https://dato.com/products/create/). In contrast to
    previous releases, we trained only one classifier using approximately 15
    million human derived variants (newly extracted from EPO 6 primate
    alignments v75) versus approximately 15 million (newly) simulated variants.
    We used an L2 penalty of 1.0 and terminated training after 10 iterations.

    nb_steps_training = 100000 # 34693009 / batch_size = 542078.265625
    nb_steps_prediction = 50000 # 350051 / batch_size = 5469.546875
    """

    # Compile the model
    model.compile(
        optimizer=optimizer, loss=loss, metrics=metrics)

    history = model.fit_generator(
        training_generator, steps_per_epoch=steps_training,
        epochs=arg_num_epochs, shuffle=True,
        validation_data=validation_generator,
        validation_steps=steps_validation)

    return(model, history)


def test(model, arg_batch_size, testing_batches_folder, steps_testing):
    testing_generator = load_batch(testing_batches_folder, arg_batch_size)
    score = model.evaluate_generator(
            testing_generator, steps=steps_testing, max_queue_size=10)
    return(score)


def save(model, model_file):
    model.save(model_file)