"""Functions to load the data
"""
import dask.dataframe as dd
from kipoi.data import BaseDataLoader, BatchDataset
from kipoi_cadd.config import get_data_dir, get_package_dir
from sklearn.model_selection import train_test_split
import pyarrow as pa
import lmdb
import pickle
import pandas as pd
import sys
from kipoi_cadd.utils import load_pickle
import logging
from tqdm import tqdm, trange
import time
import numpy as np
import blosc
import gin
import pyarrow as pa
from time import sleep
from random import randint
from torch.utils.data import DataLoader
import abc
from kipoi.data_utils import (numpy_collate, numpy_collate_concat, get_dataset_item,
                              DataloaderIterable, batch_gen, get_dataset_lens, iterable_cycle)


class Dataset(BaseDataLoader):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __getitem__(self, index):
        """Return one sample

        index: {0, ..., len(self)-1}
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self):
        """Return the number of all samples
        """
        raise NotImplementedError

    def _batch_iterable(self, batch_size=32, shuffle=False, num_workers=0, drop_last=False, **kwargs):
        """Return a batch-iteratrable

        See batch_iter docs

        Returns:
            Iterable
        """
        dl = DataLoader(self,
                        batch_size=batch_size,
                        collate_fn=numpy_collate,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        drop_last=drop_last,
                        **kwargs)
        return dl

    def batch_iter(self, batch_size=32, shuffle=False, num_workers=0, drop_last=False, **kwargs):
        """Return a batch-iterator

        Arguments:
            dataset (Dataset): dataset from which to load the data.
            batch_size (int, optional): how many samples per batch to load
                (default: 1).
            shuffle (bool, optional): set to ``True`` to have the data reshuffled
                at every epoch (default: False).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means that the data will be loaded in the main process
                (default: 0)
            drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
                if the dataset size is not divisible by the batch size. If False and
                the size of dataset is not divisible by the batch size, then the last batch
                will be smaller. (default: False)

        Returns:
            iterator
        """
        dl = self._batch_iterable(batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  drop_last=drop_last,
                                  **kwargs)
        return iter(dl)

    def load_all(self, batch_size=32, **kwargs):
        """Load the whole dataset into memory
        Arguments:
            batch_size (int, optional): how many samples per batch to load
                (default: 1).
        """
        return numpy_collate_concat([x for x in tqdm(self.batch_iter(batch_size, **kwargs))])


class CaddBatchDataset(BatchDataset):
    def __init__(self, lmbd_dir,
                 batch_idx_file, version="1.3"):

        self.version = version

        # indexed by location
        self.batch_idxs = load_pickle(batch_idx_file)
        
        self.lmdb_cadd_path = lmbd_dir
        self.lmdb_cadd = None
        self.txn = None

    def __del__(self):
        if self.lmdb_cadd is not None:
            self.lmdb_cadd.close()

    def __len__(self):
        return len(self.batch_idxs)

    def __getitem__(self, idx):
        if self.lmdb_cadd is None:
            self.lmdb_cadd = lmdb.open(self.lmdb_cadd_path, readonly=True, lock=False)
            self.txn = self.lmdb_cadd.begin()

        batch_id = self.batch_idxs[idx]
        buf = bytes(self.txn.get(batch_id.encode('ascii')))

        # Decompress and deserialize
        decom = blosc.decompress(buf)
        batch = pa.deserialize(decom)

        return batch


@gin.configurable
class CaddDataset(Dataset):
    def __init__(self, lmbd_dir,
                 variant_id_file, version="1.3"):

        self.version = version

        self.lmdb_cadd_path = lmbd_dir
        self.lmdb_cadd = None
        self.txn = None
        
        # indexed by location
        self.variant_ids_file = variant_id_file
        self.variant_ids = load_pickle(self.variant_ids_file)
        self.variant_ids = self.variant_ids.values

    def __len__(self):
        return len(self.variant_ids)

    def __del__(self):
        if self.lmdb_cadd:
            self.lmdb_cadd.close()

    def __getitem__(self, idx):
        if self.lmdb_cadd is None:
            self.lmdb_cadd = lmdb.open(self.lmdb_cadd_path, readonly=True, lock=False)
            self.txn = self.lmdb_cadd.begin()

        # TODO: Make this distiction clearer, do we want to search by loc or iloc?
        # Loc breaks when a splitted dataset doesn't have idx.
        # variant_id = self.variant_ids.loc[idx]
        # print("Getting variant ids", idx)
        variant_id = self.variant_ids[idx]
        # print("Getting the bytes for ", idx)
        buf = bytes(self.txn.get(variant_id.encode('ascii')))

        # print("Deserializing", idx)
        item = pa.deserialize(buf)
        item['targets'] = 0 if item['targets'] == -1 else item['targets']
        # TODO - check that this is not too harmful
        if np.isinf(item['inputs']).any():
            col = np.argmax(item['inputs'])
            # print("Inf number found!! idx:", idx, "var_id:", item['metadata']['variant_id'], "col:", col)
            item['inputs'] = np.minimum(item['inputs'],  65500)

        return item


    def load_all(self, batch_size=64, num_workers=64, shuffle=True, drop_last=False):
        it = self.batch_iter(batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last)
        dataset = numpy_collate_concat([x for x in tqdm(it)])
        return dataset['inputs'], dataset['targets']
    
    
    def load_all_with_metadata(self, batch_size=64, num_workers=64, shuffle=True, drop_last=False):
        it = self.batch_iter(batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last)
        return numpy_collate_concat([x for x in tqdm(it)])


def train_test_split_indexes(variant_id_file, test_size, random_state=1):
    variants = load_pickle(variant_id_file)
    train_vars, test_vars = train_test_split(
        variants, test_size=test_size, random_state=random_state)
    return train_vars, test_vars


@gin.configurable
def cadd_train_valid_data(lmdb_dir, train_id_file, valid_id_file):
    return CaddDataset(lmdb_dir, train_id_file), CaddDataset(lmdb_dir, valid_id_file)


@gin.configurable
def sparse_cadd_dataset(sparse_matrix_file, split=0.3, random_state=42):
    from scipy.sparse import load_npz
    sparse_matrix = load_npz(sparse_matrix_file)
    y = sparse_matrix[:,0]
    X = sparse_matrix[:,1:]
    print("Retrieved y", y.shape, "and X", X.shape)
    del sparse_matrix

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=split, random_state=random_state)
    return (X_train, y_train), (X_valid, y_valid)

def cadd_serialize_string_row(row, variant_id, separator, dtype=np.float16, target_col=0):
    row = np.array(row.split(separator), dtype=dtype)
    data = {"inputs": np.concatenate([row[:target_col], row[target_col + 1:]]),
        "targets": row[target_col],
        "metadata": {"variant_id": variant_id}}
    return pa.serialize(data)


def cadd_serialize_numpy_row(row, variant_id, dtype=np.float16, target_col=0):
    row = row.astype(dtype)
    data = {"inputs": np.concatenate([row[:target_col], row[target_col + 1:]]),
        "targets": row[target_col],
        "metadata": {"variant_id": variant_id}}
    return pa.serialize(data)


def cadd_deserialize_bytes(bytes_row):
    variant_info = pa.deserialize(bytes_row)
    return variant_info


def cadd_training(version='1.3'):
    return dd.read_csv(f'/s/project/kipoi-cadd/data/v{version}/training_data.tsv.gz')


#cd = CaddDataset("/s/project/kipoi-cadd/data/raw/v1.3/training_data/sample_variant_ids.pkl")
#cd.__get_n_items__(cd.df_index.sample(10))
# cadd_train_test_data("/s/project/kipoi-cadd/data/raw/v1.3/training_data/sample_variant_ids.pkl")
