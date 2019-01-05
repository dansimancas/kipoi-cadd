import argparse
import datetime
import logging
import os
import pickle
import time
from functools import reduce
import train_with_keras
import train_with_scikit


if __name__ == '__main__':
    start = time.time()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # handler = logging.StreamHandler()
    handler = logging.FileHandler(
        "/s/project/kipoi-cadd/logs/training" +
        datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".log")
    handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.info("Started script.")

    parser = argparse.ArgumentParser(description="Keras logistic regression \
    model for binary classification. This model uses a generator to load \
    batches of data.")
    parser.add_argument('library', help="Select the library to train LR. It \
    can be one of: keras, scikit, dask-scikit")
    parser.add_argument('batch_size', type=int,
                        help="Number of examples per batch.")
    parser.add_argument('num_epochs', type=int, help="Number of epochs.")
    parser.add_argument('--num_examples', type=int, help="Total number of \
    examples in the training set. This number will be further divided into \
    train, test, and validation as indicated in `split`. Only makes sense \
    when `split` is given.")
    parser.add_argument('--splits', help="Comma separated list of percentage \
    splits for train, test and validation data. Default: 80.0,10.0,10.0 to \
    split the number given in `num_examples` in chunks of 80, 10 and 10 \
    percent respectively.")
    parser.add_argument('--steps_training', type=int, help="Integer or None. \
    Total number of steps (batches of samples) before declaring one epoch \
    finished and starting the next epoch. When training with input tensors \
    such as TensorFlow data tensors, the default None is equal to the number \
    of samples in your dataset divided by the batch size, or 1 if that cannot \
    be determined.")
    parser.add_argument('--steps_testing', type=int, help="Integer or None. \
    Total number of steps (batches of samples) before declaring one epoch \
    finished and starting the next epoch.")
    parser.add_argument('--steps_validation', type=int, help="Integer or None.\
     Total number of steps (batches of samples) before declaring one epoch \
    finished and starting the next epoch.")
    parser.add_argument('--loss', help="Any type of loss accepted by \
    `keras.Models`")
    parser.add_argument('--metrics', help="Comma separated list, e.g. A,B,C")
    args = parser.parse_args()

    args_str = reduce(
        (lambda k, v: k + v),
        ["\n\t- " + k + ": " + str(v) for k, v in vars(args).items()]
        )
    logger.info("\n\tGot parameters:" + args_str)

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
        "shuffle_splits/testing/")

    splits = [float(s) for s in args.splits.split(',')] if \
        args.splits else [80, 10, 10]

    steps_training = args.steps_training if \
        args.steps_training else args.num_examples // splits[0]

    steps_testing = args.steps_testing if \
        args.steps_testing else args.num_examples // splits[1]

    steps_validation = args.steps_validation if \
        args.steps_validation else max(
            args.num_examples // splits[2],
            args.num_examples - (steps_training + steps_testing))


    if not args.library or args.library == "keras":
        model_file = ("/s/project/kipoi-cadd/data/models/" + str(args.library)
            + datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%S") +
            ".hdf5")

        # Train model
        logger.info("Started fitting model.")
        model, _ = train_with_keras.train(args.loss, args.metrics,
            args.batch_size, args.num_epochs, training_batches_folder,
            validation_batches_folder, steps_training, steps_validation)
        logger.info("Finished fitting model.")

        # Save trained model
        train_with_keras.save(model, model_file)
        logger.info("Dumped model at " + model_file + ".")

        logger.info("Evaluating model.")
        score = train_with_keras.test(model, args.batch_size,
            testing_batches_folder, steps_testing)
        logger.info('Test score: ' + str(score[0]))
        logger.info('Test accuracy: ' + str(score[1]))

    elif args.library == "scikit":
        clf_file = ("/s/project/kipoi-cadd/data/models/" + str(args.library)
            + datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%S") +
            ".joblib")
        # Train model
        logger.info("Started fitting model.")
        clf = train_with_scikit.train(args.batch_size, args.num_epochs,
            training_batches_folder, steps_training)
        logger.info("Finished fitting model.")

        # Save trained model
        train_with_scikit.save(clf, clf_file)
        logger.info("Dumped model at " + clf_file + ".")

        logger.info("Evaluating model.")
        score = train_with_keras.test(clf, args.batch_size,
            testing_batches_folder, steps_testing)
        logger.info('Test score: ' + str(score[0]))
        logger.info('Test accuracy: ' + str(score[1]))

    elif args.library == "dask-scikit":
        pass
    else:
        raise ValueError("Library must be one of: keras, scikit or \
        dask-scikit.")

    end = time.time()
    logger.info("Total elapsed time: {:.2f} minutes.".format(
        (end - start) / 60))
