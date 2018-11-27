"""Functions to load the data
"""
import dask.dataframe as dd
from kipoi.data import BatchDataset
import pyarrow as pa
import lmdb
import pickle
import numpy as np


class CaddDataset(BatchDataset):
    def __init__(self, 
                 variant_ids, version="1.3", exclude_idx=None,
                 include_idx=None):

        self.version = version

        # indexed by location
        self.df_index = load_variant_ids(variant_ids)

        # TODO - filter df_index by exclude_idx or include_idx
        if include_idx is not None:
            self.df_index = self.df_index.loc[include_idx]
        if exclude_idx is not None:
            self.df_index = self.df_index.loc[~exclude_idx]

        self.lmdb_cadd_path = "my_path{version}.lmdb"
        self.lmdb_cadd = None

    def __len__(self):
        return len(self.df_index)

    def __getitem__(self, idx):
        if self.lmdb_cadd is None:
            self.lmdb_cadd = lmdb.Environment(self.lmdb_cadd_path)

        variant_id = self.df_index.loc[idx]
        with self.lmdb_cadd.begin(write=False, buffers=True) as txn:
            buf = bytes(txn.get(variant_id.encode('ascii')))

        return pa.deserialize(buf)


def load_variant_ids(filename):
    with open(filename, 'r') as f:
        rn = pickle.load(f)
    return rn


def cadd_training_set():
    # return (train, valid) datasets
    return CaddDataset(
        exclude_idx=valid_idx), CaddDataset(include_idx=valid_idx)


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


def create_lmdb(inputfile, output_lmdb_file, variant_ids):
    varids = load_variant_ids(variant_ids)
    map_size = varids.shape[0] * 10e10
    print(map_size)
    env = lmdb.Environment(lmdbpath , map_size=20e10, max_dbs=0, lock=False)

    with env.begin(write=True, buffers=True) as txn:
        with open(inputfile) as input_file:
            _ = next(input_file)  # skip header line
            row_number = 0
            for row in tqdm(input_file):
                row = np.array(row.split(separator))

                data = {"inputs": row[1:],
                    "targets": row[0],
                    "metadata": {"row_idx": row_number}}

                buf = pa.serialize(data).to_buffer()
                txn.put(data['metadata']['variant_id'].encode('ascii'), buf)

                row_number += 1

                if row_number > 10: break


def cadd_training(version='1.3'):
    return dd.read_csv(f'/s/project/kipoi-cadd/data/v{version}/training_data.tsv.gz')
