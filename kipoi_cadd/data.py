"""Functions to load the data
"""
import dask.dataframe as dd
from kipoi.data import BatchDataset
from kipoi_cadd.config import get_data_dir, get_package_dir
import pyarrow as pa
import lmdb
import pickle
import pandas as pd
import sys
import logging
from tqdm import tqdm
import time
import numpy as np
import gin.tf

# @gin.configurable
class CaddDataset(BatchDataset):
    def __init__(self, 
                 variant_ids, version="1.3", exclude_idx=None,
                 include_idx=None, separator=',', map_size=83904000):

        self.version = version

        # indexed by location
        self.df_index = load_variant_ids(variant_ids)

        # TODO - filter df_index by exclude_idx or include_idx
        if include_idx is not None:
            self.df_index = self.df_index.loc[include_idx]
        if exclude_idx is not None:
            bad_df = self.df_index.index.isin(exclude_idx)
            self.df_index = self.df_index[~bad_df]

        self.lmdb_cadd_path = get_data_dir() + f"/raw/v{version}/training_data/lmdb"
        # self.lmdb_cadd_path = get_data_dir() + "/tests/lmdb_3/"
        self.lmdb_cadd = None
        self._map_size = map_size

    def __len__(self):
        return len(self.df_index)

    def __getitem__(self, idx):
        if self.lmdb_cadd is None:
            self.lmdb_cadd = lmdb.Environment(self.lmdb_cadd_path, map_size=self._map_size, lock=False)

        variant_id = self.df_index.loc[idx]
        with self.lmdb_cadd.begin(write=False, buffers=True) as txn:
            buf = bytes(txn.get(variant_id.encode('ascii')))

        return pa.deserialize(buf)

    def get_n_items(self, var_idxs):
        if self.lmdb_cadd is None:
            self.lmdb_cadd = lmdb.Environment(self.lmdb_cadd_path, map_size=self._map_size, lock=False)

        items_df = None

        with self.lmdb_cadd.begin(write=False, buffers=True) as txn:
            for var in var_idxs:
                buf = bytes(txn.get(var.encode('ascii')))
                desbuf = pa.deserialize(buf)
                data = np.insert(desbuf['inputs'], 0, desbuf['targets'])
                if items_df is None:
                    items_df = pd.DataFrame([data], index=[desbuf['metadata']['variant_id']])
                else:
                    items_df = items_df.append(pd.DataFrame([data], index=[desbuf['metadata']['variant_id']]))
        return items_df


def load_variant_ids(filename):
    with open(filename, 'rb') as f:
        rn = pickle.load(f)
    return rn

# @gin.configurable
def cadd_train_test_data(lmdb_file, train_id_file, validd_id_file):
    df_index = load_variant_ids(variant_ids)
    train_idx = df_index.sample(frac=0.9).index.values
    return CaddDataset(variant_ids, include_idx=train_idx), CaddDataset(variant_ids, exclude_idx=train_idx)


def cadd_serialize_string_row(row, variant_id, separator, dtype=np.float16, target_col=0):
    import pyarrow as pa
    row = np.array(row.split(separator), dtype=dtype)
    data = {"inputs": np.concatenate([row[:target_col], row[target_col + 1:]]),
        "targets": row[target_col],
        "metadata": {"variant_id": variant_id}}
    return pa.serialize(data)


def cadd_serialize_numpy_row(row, variant_id, separator, dtype=np.float16, target_col=0):
    import pyarrow as pa
    row = row.astype(dtype)
    data = {"inputs": np.concatenate([row[:target_col], row[target_col + 1:]]),
        "targets": row[target_col],
        "metadata": {"variant_id": variant_id}}
    return pa.serialize(data)


def create_lmdb(inputfile=get_data_dir() + "/raw/v1.3/training_data/training_data.imputed.csv",
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
        row_example.values[0], varids[0], separator,
        np.float16, 0).to_buffer().size

    multipl = 1.9
    map_size = int(map_size * varids.shape[0] * multipl)

    logger.info("Using map_size: " + str(map_size) + ". The multiplier applied was " + str(multipl))
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


def cadd_training(version='1.3'):
    return dd.read_csv(f'/s/project/kipoi-cadd/data/v{version}/training_data.tsv.gz')


#cd = CaddDataset("/s/project/kipoi-cadd/data/raw/v1.3/training_data/sample_variant_ids.pkl")
#cd.__get_n_items__(cd.df_index.sample(10))
# cadd_train_test_data("/s/project/kipoi-cadd/data/raw/v1.3/training_data/sample_variant_ids.pkl")