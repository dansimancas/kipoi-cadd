import pandas as pd
import os
import argparse
import time
import datetime
import pickle
import logging
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense


def load_batch(directory, batch_size, sep=','):
    sub_batch = None
    # For keras, my generator must be able to loop infinitely
    while True:
        for file in os.listdir(directory):
            filename = directory + str(os.fsdecode(file))
            rows_df = pd.read_csv(filename, sep=sep)
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


if __name__ == '__main__':
    start = time.time()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # handler = logging.StreamHandler()
    handler = logging.FileHandler(
        "/s/project/kipoi-cadd/logs/training" +
        datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%S") + ".log")
    handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.info("Started script.")

    parser = argparse.ArgumentParser(description="Keras logistic regression \
    model for binary classification. This model uses a generator to load \
    batches of data.")
    parser.add_argument('batch_size', type=int,
                        help="Number of examples per batch.")
    parser.add_argument('num_epochs', type=int, help="Number of epochs.")
    parser.add_argument('num_examples', type=int, help="Total number of \
    examples in the training set. This number will be further divided into \
    train, test, and validation as indicated in `split`.")
    parser.add_argument('split', help="Comma separated list of percentage \
    splits for train, test and validation data. Default: 80,10,10 to split \
    the number given in `num_examples` in chunks of 80, 10 and 10 percent \
    respectively.")
    parser.add_argument('--steps_training', type=int, help="Integer or None. Total \
    number of steps (batches of samples) before declaring one epoch finished \
    and starting the next epoch. When training with input tensors such as \
    TensorFlow data tensors, the default None is equal to the number of \
    samples in your dataset divided by the batch size, or 1 if that cannot \
    be determined.")
    parser.add_argument('--steps_testing', type=int, help="Integer or None. Total \
    number of teps (batches of samples) before declaring one epoch finished \
    and starting the next epoch.")
    parser.add_argument('--steps_validation', type=int, help="Integer or None. Total \
    number of teps (batches of samples) before declaring one epoch finished \
    and starting the next epoch.")
    parser.add_argument('--loss', help="Any type of loss accepted by \
    `keras.Models`")
    parser.add_argument('--metrics', help="Comma separated list, e.g. A,B,C")
    args = parser.parse_args()

    logger.info("\n\tGot parameters:" + args)

    # Initialize variables
    training_imputed = ("/s/project/kipoi-cadd/data/raw/v1.3/training_data/" +
                        "training_data.imputed.csv")
    training = ("/s/project/kipoi-cadd/data/raw/v1.3/training_data/" +
                "training_data.tsv")
    training_batches_folder = (
        "/s/project/kipoi-cadd/data/raw/v1.3/training_data/" +
        "shuffle_splits/training/")
    testing_batches_folder = (
        "/s/project/kipoi-cadd/data/raw/v1.3/training_data/" +
        "shuffle_splits/testing/")
    validation_batches_folder = (
        "/s/project/kipoi-cadd/data/raw/v1.3/training_data/" +
        "shuffle_splits/validation/")
    model_file = ("/s/project/kipoi-cadd/data/models/model" +
                  datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%S") +
                  ".pickle")

    splits = [int(s) for s in args.splits.split(',')] if \
        args.splits else [80, 10, 10]

    steps_training = args.steps_training if \
        args.steps_training else args.num_examples // splits[0]

    steps_testing = args.steps_testing if \
        args.steps_testing else args.num_examples // splits[1]

    steps_validation = args.steps_validation if \
        args.steps_validation else max(
            args.num_examples // splits[2],
            args.num_examples - (steps_training + steps_testing))

    loss = 'binary_crossentropy' if args.loss is None else args.loss
    metrics = ['accuracy'] if args.metrics is None else args.metrics.split(',')

    # Build the model
    output_dim = 1  # One binary class
    input_dim = 1063  # number of features of the input
    model = Sequential()
    model.add(Dense(output_dim, input_dim=input_dim, activation='sigmoid',
                    kernel_regularizer=l2(1.0)))

    training_generator = load_batch(training_batches_folder, args.batch_size)
    testing_generator = load_batch(testing_batches_folder, args.batch_size)
    validation_generator = load_batch(validation_batches_folder,
                                      args.batch_size)

    """
    *CADD v1.3 Release Notes*
    Learner: For this version we used the Logistic Regression module of
    GraphLab Create v1.4 (https://dato.com/products/create/). In contrast to
    previous releases, we trained only one classifier using approximately 15
    million human derived variants (newly extracted from EPO 6 primate
    alignments v75) versus approximately 15 million (newly) simulated variants.
    We used an L2 penalty of 1.0 and terminated training after 10 iterations.

    nb_steps : 35043060 / 3200 = 10951
    """

    # Compile the model
    model.compile(
        optimizer='sgd', loss=loss, metrics=metrics)

    logger.info("Started fitting model.")

    history = model.fit_generator(
        training_generator, steps_per_epoch=steps_training,
        epochs=args.num_epochs, shuffle=True,
        validation_data=validation_generator,
        validation_steps=steps_validation)

    logger.info("Finished fitting model.")

    # Save trained model
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
        logger.info("Dumped model at " + model_file + ".")

    logger.info("Evaluating model.")
    score = model.evaluate_generator(
        testing_generator, steps=steps_testing, max_queue_size=10)
    logger.info('Test score:', score[0])
    logger.info('Test accuracy:', score[1])

    end = time.time()
    logger.info("Total elapsed time: {:.2f} minutes.".format(
        (end - start) / 60))
