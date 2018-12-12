"""Test lmdb functionality
"""
from gin_train.trainers import KerasTrainer
import lmdb
import pandas as pd
import pyarrow as pa
import numpy as np
from kipoi_cadd.config import get_data_dir
import shutil
import pickle
import logging
from kipoi_cadd.data import cadd_serialize_string_row
from kipoi_cadd.data import cadd_serialize_numpy_row
from kipoi_cadd.data import cadd_train_test_valid_data
from kipoi_cadd.data import cadd_train_test_data
from kipoi_cadd.data import save_variant_ids
from keras.models import load_model
from kipoi_cadd.data import train_test_split_indexes
from sklearn.model_selection import train_test_split
from kipoi_cadd.data import load_variant_ids
from kipoi_cadd.models import logistic_regression_scikit
from kipoi_cadd.models import logistic_regression_keras
from gin_train.metrics import ClassificationMetrics, auprc
from kipoi_cadd.utils import load_pickle
import os
import time
from tqdm import tqdm
import sys
import datetime


def test_lmdb_get_put():
    ddir = get_data_dir()
    X = pd.read_csv(ddir + "/raw/v1.3/training_data/shuffle_splits/testing/3471.csv", index_col=0)
    Xnp = np.array(X)
    print("Nbytes:", Xnp.nbytes)
    lmdbpath = ddir + "/tests/lmdb_1"
    env = lmdb.Environment(lmdbpath, map_size=Xnp.nbytes*6, max_dbs=0, lock=False)
    check_index = X.index[10]

    with env.begin(write=True, buffers=True) as txn: 
        for i in range(X.shape[0]):
            data = {"inputs": X.iloc[i, 1:],
                    "targets": X.iloc[i, 0],
                    "metadata": {"variant_id": "bla",
                                "row_idx": str(X.index[i])}}

            buf = pa.serialize(data).to_buffer()
            # db.put(key, value)
            # print(data['metadata']['row_idx'])
            txn.put(data['metadata']['row_idx'].encode('ascii'), buf)

    with env.begin(write=False, buffers=True) as txn:
        buf = txn.get(str(check_index).encode('ascii'))  # only valid until the next write.
        buf_copy = bytes(buf)       # valid forever

    variant = pa.deserialize(buf_copy)
    s = X.loc[check_index]

    # if os.path.exists(lmdbpath):
    #    shutil.rmtree(lmdbpath, ignore_errors=True)
    #    os.rmdir(lmdbpath)

    assert variant['inputs'].equals(s[1:])


def test_lmdb_get_put_with_variant_id():
    ddir = get_data_dir()
    with open(ddir + "/raw/v1.3/training_data/sample_variant_ids.pkl", 
              'rb') as f:
        varids = pickle.load(f)
    inputfile = \
        get_data_dir() + "/raw/v1.3/training_data/training_data.imputed.csv"
    separator = ','
    choose = 8
    lmdbpath = ddir + "/tests/lmdb_2"

    rows = {"variant_ids": [], "row_infos": []}
    with open(inputfile) as input_file:
        _ = next(input_file)  # skip header line
        row_number = 0
        row_example = None
        for row in tqdm(input_file):
            row = np.array(row.split(separator), dtype=np.float16)
            rows["variant_ids"].append(varids.iloc[row_number, 0])
            rows["row_infos"].append(row)
            row_number += 1
            if row_number > 10:
                row_example = (row, varids.iloc[row_number, 0])
                break
            
    map_size = cadd_serialize_string_row(row_example[0], row_example[1], separator,
                                  np.float16, 0).to_buffer().size
    map_size = map_size * varids.shape[0] * 1.2
    env = lmdb.Environment(lmdbpath , map_size=map_size, max_dbs=0, lock=False)

    with env.begin(write=True, buffers=True) as txn:
        with open(inputfile) as input_file:
            _ = next(input_file)  # skip header line
            row_number = 0
            for row in tqdm(input_file):
                variant_id = varids.iloc[row_number, 0]
                ser_data = cadd_serialize_string_row(
                    row, variant_id, separator, np.float16, 0)

                buf = ser_data.to_buffer()
                print(buf.size)
                txn.put(variant_id.encode('ascii'), buf)

                row_number += 1

                if row_number > 10: break

    
    find_variant = varids.iloc[choose, 0]
    print("Find variant", find_variant)
    with env.begin(write=False, buffers=True) as txn:
        buf = bytes(txn.get(find_variant.encode('ascii')))

    variant_info = pa.deserialize(buf)['inputs']
    check_variant_info = np.array(rows["row_infos"][choose][1:])

    # if os.path.exists(lmdbpath):
    #    shutil.rmtree(lmdbpath, ignore_errors=True)
    #    os.rmdir(lmdbpath)

    assert np.array_equal(variant_info, check_variant_info)


def test_put_10000_variants():
    start = time.time()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(
        "/s/project/kipoi-cadd/logs/lmdb/put.log")
    handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

    ddir = "/s/project/kipoi-cadd/data"
    variantidpath = ddir + "/raw/v1.3/training_data/sample_variant_ids.pkl"

    with open(variantidpath, 'rb') as f:
        varids = pickle.load(f)

    inputfile = (get_data_dir() +
        "/raw/v1.3/training_data/training_data.imputed.csv")
    separator = ','
    lmdbpath = ddir + "/tests/lmdb_3/"

    logger.info(("Started script. Parameters are:\n\t" +
                 "inputfile: " + inputfile + "\n\t" + 
                 "lmdbpath: " + lmdbpath + "\n\t" + 
                 "separator: '" + separator + "'\n\t" + 
                 "variantidpath: " + variantidpath
                 ))

    row_example = pd.read_csv(inputfile,
                              sep=separator,
                              nrows=1,
                              skiprows=1,
                              header=None)
    map_size = cadd_serialize_numpy_row(
        row_example.values[0], varids[0],
        np.float16, 0).to_buffer().size

    multipl = 1.9
    map_size = int(map_size * varids.shape[0] * multipl)

    logger.info("Using map_size: " + str(map_size) + ". The multiplier applied was " + str(multipl))
    env = lmdb.Environment(lmdbpath , map_size=map_size, max_dbs=0, lock=False)

    with env.begin(write=True, buffers=True) as txn:
        with open(inputfile) as input_file:
            _ = next(input_file)  # skip header line
            row_number = 0
            for row in tqdm(input_file):
                variant_id = varids[row_number]
                ser_data = cadd_serialize_string_row(
                    row, variant_id, separator, np.float16, 0)

                buf = ser_data.to_buffer()
                try:
                    txn.put(variant_id.encode('ascii'), buf)
                except lmdb.MapFullError as err:
                    logger.error(str(err) + ". Exiting the program.")
                    sys.exit()
                row_number += 1
                if row_number >= 10000: break

    logger.info("Finished putting " + str(row_number) + " rows to lmdb.")
    end = time.time()
    logger.info("Total elapsed time: {:.2f} minutes.".format(
        (end - start) / 60))


def test_get_10000_variants():
    ddir = "/s/project/kipoi-cadd/data"
    with open(ddir + "/raw/v1.3/training_data/sample_variant_ids.pkl", 
              'rb') as f:
        varids = pickle.load(f)
    varids = varids.sample(len(varids))
    lmdbpath = ddir + "/tests/lmdb_3"
    env = lmdb.Environment(lmdbpath, lock=False)
    num_vars = 0
    with env.begin(write=False, buffers=True) as txn:
        for var in varids:
            bytes(txn.get(var.encode('ascii')))
            num_vars += 1
    print("Finished getting", num_vars, "rows.")


def test_split_train_test():
    variant_ids = "/s/project/kipoi-cadd/data/raw/v1.3/training_data/sample_variant_ids.pkl"
    lmdb_dir = "/s/project/kipoi-cadd/data/raw/v1.3/training_data/lmdb/"
    train_idx, test_idx = train_test_split_indexes(variant_ids, test_size=0.3)
    training_data_dir = "/s/project/kipoi-cadd/data/raw/v1.3/training_data/"
    lmdb_dir = training_data_dir + "lmdb"
    save_variant_ids(training_data_dir + "train_idx.pkl", train_idx)
    save_variant_ids(training_data_dir + "test_idx.pkl", test_idx)
    train, test = cadd_train_test_data(
        lmdb_dir, training_data_dir + "train_idx.pkl", training_data_dir + "test_idx.pkl")


def test_split_train_logistic_regression_with_generator():
    variant_ids = "/s/project/kipoi-cadd/data/raw/v1.3/training_data/sample_variant_ids.pkl"
    train_idx, test_idx = train_test_split_indexes(variant_ids, test_size=0.3)
    training_data_dir = "/s/project/kipoi-cadd/data/raw/v1.3/training_data/"
    lmdb_dir = training_data_dir + "lmdb"
    save_variant_ids(training_data_dir + "train_idx.pkl", train_idx)
    save_variant_ids(training_data_dir + "test_idx.pkl", test_idx)
    train, test = cadd_train_test_data(
        lmdb_dir, training_data_dir + "train_idx.pkl", training_data_dir + "test_idx.pkl")
    
    clf = logistic_regression_scikit
    iterator = train.batch_iter(batch_size=32, num_workers=1)
    count = 0
    for b in iterator:
        count += 1
    print(count)


def test_update_targets():
    variant_ids = "/s/project/kipoi-cadd/data/raw/v1.3/training_data/sample_variant_ids.pkl"
    varids = load_variant_ids(variant_ids)
    lmdb_dir = "/s/project/kipoi-cadd/data/tests/lmdb_3/"
    num_vars=0
    inputfile = \
        get_data_dir() + "/raw/v1.3/training_data/training_data.imputed.csv"

    row_example = pd.read_csv(inputfile,
                              sep=',',
                              nrows=1,
                              skiprows=1,
                              header=None)
    map_size = cadd_serialize_numpy_row(
        row_example.values[0], varids[0],
        np.float16, 0).to_buffer().size
    map_size = map_size * (varids.shape[0] + 1) * 5

    env = lmdb.Environment(lmdb_dir, lock=False, map_size=map_size, writemap=True)
    with env.begin(write=True, buffers=True) as txn:
        for var in varids:
            row = bytes(txn.get(var.encode('ascii')))
            np_row = pa.deserialize(row)
            if np_row['targets'] == -1:
                np_row['targets'] = 0
                ser_data = pa.serialize(np_row)
                buf = ser_data.to_buffer()
                txn.replace(var.encode('ascii'), buf)
                num_vars += 1
    print("Finished changing", num_vars, "rows.")
    

def test_train_batch():
    training_data_dir = "/s/project/kipoi-cadd/data/raw/v1.3/training_data/"
    lmdb_dir = training_data_dir + "lmdb"
    train_idx_path = training_data_dir + "train_idx.pkl"
    test_idx_path = training_data_dir + "test_idx.pkl"
    valid_idx_path = training_data_dir + "valid_idx.pkl"

    train, test, valid = cadd_train_test_valid_data(lmdb_dir, train_idx_path, test_idx_path, valid_idx_path)
    tr = KerasTrainer(logistic_regression_keras(n_features=1063), train, valid, get_data_dir() + "/models/try5")
    tr.train(batch_size=512, epochs=50, num_workers=1)


def test_evaluate_model():
    model_dir = get_data_dir() + "/models/try5/model.h5"
    training_data_dir = "/s/project/kipoi-cadd/data/raw/v1.3/training_data/"
    lmdb_dir = training_data_dir + "lmdb"
    train_idx_path = training_data_dir + "train_idx.pkl"
    test_idx_path = training_data_dir + "test_idx.pkl"
    valid_idx_path = training_data_dir + "valid_idx.pkl"

    tr = load_model(model_dir)
    _, _, valid = cadd_train_test_valid_data(lmdb_dir, train_idx_path, test_idx_path, valid_idx_path)
    metric = ClassificationMetrics()

    tr.evaluate(metric(valid))


def test_train_and_evaluate():
    training_data_dir = "/s/project/kipoi-cadd/data/raw/v1.3/training_data/"
    lmdb_dir = training_data_dir + "lmdb"
    train_idx_path = training_data_dir + "train_idx.pkl"
    test_idx_path = training_data_dir + "test_idx.pkl"
    valid_idx_path = training_data_dir + "valid_idx.pkl"

    train, test, valid = cadd_train_test_valid_data(lmdb_dir, train_idx_path, test_idx_path, valid_idx_path)
    tr = KerasTrainer(logistic_regression_keras(n_features=1063), train, valid, get_data_dir() + "/models/try6")
    tr.train(batch_size=512, epochs=50, num_workers=1)

    metric = ClassificationMetrics()

    tr.evaluate(metric)

def test_pyarrow_serialization():
    pass