from kipoi.data import BaseDataLoader, BatchDataset
from kipoi_cadd.config import get_data_dir, get_package_dir
from sklearn.model_selection import train_test_split
import pyarrow as pa
import lmdb
import pickle
import pandas as pd
import sys
from kipoi_cadd.utils import load_pickle, OrderedSet
import logging
from tqdm import tqdm
from kipoi_cadd.data import cadd_serialize_numpy_row, cadd_serialize_string_row
import time
from collections import OrderedDict
import numpy as np
import blosc
import gin
import pyarrow as pa
from time import sleep
from random import randint


def create_lmdb(
    inputfile=get_data_dir() + "/raw/v1.3/training_data/training_data.imputed.csv",
    output_lmdb_file=get_data_dir() + "/raw/v1.3/training_data/lmdb",
    variant_ids=get_data_dir() + "/raw/v1.3/training_data/variant_ids.pkl",
    separator=',',
    log=get_package_dir() + "/logs/lmdb/put.log"):

    start = time.time()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log)
    handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

    logger.info(("Started script. Parameters are:\n\t" +
                 "inputfile: " + inputfile + "\n\t" + 
                 "output_lmdb_file: " + output_lmdb_file + "\n\t" + 
                 "separator: '" + separator + "'\n\t" + 
                 "variant_ids: " + variant_ids
                 ))

    with open(variant_ids, 'rb') as f:
        varids = pickle.load(f)
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

    logger.info("Using map_size: " + str(map_size) + ". The multiplier applied was " +
                str(multipl))
    env = lmdb.Environment(output_lmdb_file , map_size=map_size, max_dbs=0, lock=False)

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

    logger.info("Finished putting " + str(row_number) + " rows to lmdb.")
    end = time.time()
    logger.info("Total elapsed time: {:.2f} minutes.".format(
        (end - start) / 60))


def create_lmdb_from_iterator(
    it,
    output_lmdb_file=get_data_dir() + "/raw/v1.3/training_data/lmdb_batched",
    batch_idx_file=get_data_dir() + "/raw/v1.3/training_data/shuffle_splits/batch_idxs.pkl",
    map_size=23399354270,
    log=get_package_dir() + "/logs/lmdb/put.log"):

    start = time.time()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log)
    handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

    logger.info(("Started script. Parameters are:\n\t" +
                 "iterator: " + str(type(it)) + "\n\t" + 
                 "output_lmdb_file: " + output_lmdb_file + "\n\t" +
                 "batch_idx_file: " + batch_idx_file + "\n\t"
                 ))

    index_mapping = OrderedDict()
    map_size = None
    txn = None
    batch_num = 0

    env = lmdb.Environment(output_lmdb_file , map_size=map_size, max_dbs=0, lock=False)
    with env.begin(write=True, buffers=True) as txn:
        for batch in tqdm(it):
            index_mapping[batch_num] = {
                "variant_ids":OrderedSet(batch['metadata']['variant_id'])}

            # Serialize and compress
            buff = pa.serialize(batch).to_buffer()
            blzpacked = blosc.compress(buff, typesize=8, cname='blosclz')

            try:
                txn.put(str(batch_num).encode('ascii'), blzpacked)
            except lmdb.MapFullError as err:
                logger.error(str(err) + ". Exiting the program.")

            batch_num += 1

    logger.info("Finished putting " + str(batch_num) + " batches to lmdb.")
    end = time.time()
    logger.info("Total elapsed time: {:.2f} minutes.".format(
        (end - start) / 60))

    
def load_csv_to_sparse_matrix(csv_file, blocksize=10E6, final_type=np.float32):
    import dask.dataframe as ddf
    from scipy.sparse import csr_matrix
    from dask.diagnostics import ProgressBar
    
    print("Started dask task.")
    df_dask = ddf.read_csv(csv_file, blocksize=blocksize, assume_missing=True)
    df_dask = df_dask.map_partitions(lambda part: part.to_sparse(fill_value=0))
    
    with ProgressBar():
        df_dask = df_dask.compute().reset_index(drop=True)
    
    print("Finished dask task.")
    csr = csr_matrix(df_dask, dtype=final_type, copy=True)
    print("Finished transforming to csr_matrix.")
    print("Changing -1 in the targets")
    y_train.data[y_train.data == -1] = 0
    y_valid.data[y_valid.data == -1] = 0
    del df_dask

    return csr


def put_batches(csv_file, lmdb_batched_dir, batch_size=256, separator=','):
    with open(variant_ids, 'rb') as f:
        varids = pickle.load(f)

    row_example = pd.read_csv(csv_file,
                              sep=separator,
                              nrows=1,
                              skiprows=1,
                              header=None)

    map_size = cadd_serialize_numpy_row(
        row_example.values[0], varids[0],
        np.float16, 0).to_buffer().size

    multipl = 1.9
    map_size = int(map_size * varids.shape[0] * multipl)
    
    env = lmdb.Environment(lmdbpath, map_size=map_size, max_dbs=0, lock=False)

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

    
def get_one_batch(lmdb_batch_dir, idx):
    env = lmdb.Environment(lmdb_batch_dir, readonly=True, lock=False)
    with env.begin() as txn:
        buff = bytes(txn.get(str(idx).encode('ascii')))
        ser = blosc.decompress(buff)
        batch = pa.deserialize(ser)
    return batch


def dir_batch_generator(directory, batch_size, sep=','):
    import os

    sub_batch = None
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