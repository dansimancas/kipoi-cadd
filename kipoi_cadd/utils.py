import collections
import sys
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from kipoi_cadd.config import get_data_dir
import pickle
from collections import OrderedDict
from tqdm import trange

class OrderedSet(collections.Set):
    def __init__(self, iterable=()):
        self.d = collections.OrderedDict.fromkeys(iterable)

    def __len__(self):
        return len(self.d)

    def __contains__(self, element):
        return element in self.d

    def __iter__(self):
        return iter(self.d)


def write_json(obj, fname, **kwargs):
    with open(fname, "w") as f:
        return json.dump(obj, f, **kwargs)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        rn = pickle.load(f)
    return rn


def dump_to_pickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def generate_variant_ids(inputfile, outputfile, separator='\t'):
    input_df = pd.read_csv(inputfile,
                           sep=separator,
                           usecols=['Chrom', 'Pos', 'Ref', 'Alt'],
                           dtype={
                               'Chrom': 'str',
                               'Pos': np.int32,
                               'Ref': 'str',
                               'Alt': 'str'})
    #                        nrows=10000)
    variant_ids = input_df.apply(
        lambda row: ':'.join([row[0], str(row[1]), row[2], ("['" +
                             row[3] + "']")]), axis=1)
        
    with open(outputfile, 'wb') as f:
        pickle.dump(variant_ids, f)


def generate_batch_idxs(shuffled_idxs_file,
                        variant_id_file,
                        batch_size,
                        outputfile,
                        num_batches=None):

    shuffled_idxs = load_pickle(shuffled_idxs_file)
    variant_ids = load_pickle(variant_id_file).values
    num_indexes = len(shuffled_idxs)
    amount_batches = (num_indexes // batch_size) + 1
    num_loops = min(
        num_batches, amount_batches) if num_batches else amount_batches
    batch_indexes = OrderedDict()

    for i in trange(num_loops):
        start = (i) * batch_size
        end = min(num_indexes, start + batch_size)
        batch_indexes[i] = {
            "row_nrs": OrderedSet(shuffled_idxs[start:end]),
            "variant_ids":OrderedSet(variant_ids[shuffled_idxs[start:end]])}

    with open(outputfile, 'wb') as f:
        pickle.dump(batch_indexes, f)


if __name__ == '__main__':
    inputfile = get_data_dir() + "/raw/v1.3/training_data/training_data.tsv"
    shuffled_idxs_file = get_data_dir() + "/raw/v1.3/training_data/shuffle_splits/shuffled_index.pickle"
    variant_id_file = get_data_dir() + "/raw/v1.3/training_data/variant_ids.pkl"
    outputfile = get_data_dir() + "/raw/v1.3/training_data/shuffle_splits/batch_idxs_256.pkl"
    generate_batch_idxs(shuffled_idxs_file, variant_id_file, 256, outputfile)

