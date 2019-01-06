from kipoi.data import BaseDataLoader, BatchDataset
from kipoi_cadd.config import get_data_dir, get_package_dir
from sklearn.model_selection import train_test_split
import pyarrow as pa
import lmdb
import pickle
import pandas as pd
import sys
from kipoi_cadd.utils import load_pickle, dump_to_pickle, OrderedSet, get_all_files_extension
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


def create_batched_lmdb_from_iterator(it, lmdb_batch_dir, variant_ids_file, num_batches=-1,
                              map_size=23399354270):
    start = time.time()

    index_mapping = OrderedDict()
    map_size = None
    txn = None
    batch_num = 0
    variant_ids = load_pickle(variant_ids_file)

    env = lmdb.Environment(lmdb_batch_dir, map_size=map_size, max_dbs=0, lock=False)
    with env.begin(write=True, buffers=True) as txn:
        for batch in tqdm(it):
            b = {
                "batch_id": np.int32(batch_num),
                "inputs": batch[0].values.astype(np.float16),
                "targets": batch[1].values.astype(np.float16),
                "metadata": {
                    "row_num": np.array(batch[0].index, dtype=np.int32),
                    "variant_id": np.array(variant_ids.loc[batch[0].index], dtype='<U20')
                }
            }

            # Serialize and compress
            buff = pa.serialize(b).to_buffer()
            blzpacked = blosc.compress(buff, typesize=8, cname='blosclz')

            try:
                txn.put(str(batch_num).encode('ascii'), blzpacked)
            except lmdb.MapFullError as err:
                print(str(err) + ". Exiting the program.")

            batch_num += 1
            # if batch_num >= num_batches: break

    print("Finished putting " + str(batch_num) + " batches to lmdb.")
    end = time.time()
    print("Total elapsed time: {:.2f} minutes.".format(
        (end - start) / 60))


def calculate_map_size(row_example, nrows, multiplier=1.9):
    row_size = pa.serialize(row_example).to_buffer().size
    map_size = int(row_size * nrows * multiplier)
    return map_size
    

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

    
def load_csv_to_sparse_matrix(csv_file, targets_col=0, blocksize=10E6, final_type=np.float32):
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
    
    del df_dask
    
    print("Changing -1 in the targets...")
    y_array = csr[:, targets_col].toarray()
    y_array[y_array==-1] = 0
    new_y = csr_matrix(y_array)
    csr[:, targets_col] = new_y
    print("Done.")
    
    return csr


def load_csv_to_pandas_sparse(csv_file, output, targets_col=0, blocksize=10E6, final_type=np.float32):
    import dask.dataframe as ddf
    from scipy.sparse import csr_matrix
    from dask.diagnostics import ProgressBar
    
    print("Started dask task.")
    df_dask = ddf.read_csv(csv_file, blocksize=blocksize, assume_missing=True)
    df_dask = df_dask.map_partitions(lambda part: part.to_sparse(fill_value=0))
    
    with ProgressBar():
        df_dask = df_dask.compute().reset_index(drop=True)
    
    print("Finished dask task.")
    
    dump_to_pickle(output, df_dask)
    del df_dask

    
def load_csv_chunks_tosparse(filename, chunksize, dtype, num_lines, output=None, header=0, index_col=None):
    from scipy.sparse import csr_matrix, vstack, save_npz
    
    full_matrix = None
    tqdm_total = (num_lines//chunksize) + 1 if num_lines % chunksize > 0 else (num_lines/chunksize) - 1
    
    for chunk in tqdm(pd.read_csv(filename, chunksize=chunksize, dtype=dtype, header=header,
                                  index_col=index_col), total=tqdm_total):
        if full_matrix is None:
            full_matrix = csr_matrix(chunk, shape=chunk.shape, dtype=dtype)
        else:
            chunk_csr = csr_matrix(chunk, shape=chunk.shape, dtype=dtype)
            full_matrix = vstack([full_matrix, chunk_csr])
    
    if output is None:
        return full_matrix
    else:
        save_npz(output, full_matrix)
    

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
    import os, glob

    sub_batch = None
    
    # for file in os.listdir(directory):
    for file in get_all_files_extension(directory, ".csv"):
        filename = str(os.fsdecode(file))
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
                

def cadd_generate_batched_lmdb_from_many_csv(lmdb_batch_dir, csv_folder, variant_ids_file, batch_size, num_batches=-1):
    it = dir_batch_generator(csv_folder, batch_size)
    test_batch = next(it)
    variant_ids = load_pickle(variant_ids_file)
    nrows = len(variant_ids)
    
    row_example = {
    "batch_id": np.int32(0),
    "inputs": test_batch[0].values.astype(np.float16),
    "targets": test_batch[1].values.astype(np.float16),
    "metadata": {
        "row_num": np.array(test_batch[0].index, dtype=np.int32),
        "variant_id": np.array(variant_ids.loc[test_batch[0].index], dtype='<U20')
        }
    }
    
    ms = calculate_map_size(row_example, nrows)
    it = dir_batch_generator(csv_folder, batch_size)
    create_batched_lmdb_from_iterator(it, lmdb_batch_dir, variant_ids_file, num_batches=num_batches, map_size=ms)