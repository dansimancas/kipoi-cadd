import collections
import sys
import json
from tqdm import tqdm
import pandas as pd
from kipoi_cadd.config import get_data_dir
import pickle

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


def generate_variant_ids(inputfile, outputfile, separator='\t'):
    import numpy as np
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
        lambda row: ':'.join([str(row[0]), str(row[1]), row[2], ("['" +
                             row[3] + "']")]), axis=1)
        
    with open(outputfile, 'wb') as f:
        pickle.dump(variant_ids, f)


if __name__ == '__main__':
    inputfile = get_data_dir() + "/raw/v1.3/training_data/training_data.tsv"
    outputfile = get_data_dir() + "/raw/v1.3/training_data/variant_ids.pkl"
    generate_variant_ids(inputfile, outputfile)