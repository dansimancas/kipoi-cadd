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


def variant_id_string(chrom, pos, ref, alt, use_chr_word=False):
    if use_chr_word:
        chrom = chrom if 'chr' in chrom else 'chr' + chrom
    else:
        chrom = chrom if 'chr' not in chrom else chrom.split('chr')[1]
    
    if isinstance(alt, list):
        var_id_str = ':'.join([chrom, str(pos), ref, str(alt)])
    else:
        var_id_str = ':'.join([chrom, str(pos), ref, ("['" + alt + "']")])
    return var_id_str


def generate_intervals_file(list_variant_ids, output_file=None, sort=True,
                            use_chr_word=True, header=None):
    """ Generates an intervals file with chr, start, end columns.
    By default, the chromosome number will be preceded with the word chr.
    """
    intervals = {'chr': [], 'start': [], 'end': []}
    for var in list_variant_ids:
        chrom, pos, ref, alt = decompose_variant_string(var)
        intervals['chr'].append(chrom)
        intervals['start'].append(pos)
        intervals['end'].append(pos+1)
    df = pd.DataFrame(intervals)
    if sort: df.sort_values(by=['chr', 'start'], inplace=True)
    if use_chr_word:
        df.chr = ["chr" + str(c) for c in df.chr]
    if output_file is not None:
        df.to_csv(output_file, sep="\t", index=None, header=None)
    else:
        return df
    

def decompose_variant_string(variant_string, try_convert=True):
    """Decomposes a variant string of type 1:34345:A:['T']. It does not
    expect the chromosome number to be preceded with the word `chr`.
    """
    chrom, pos, ref, alts = variant_string.split(":")
    alts = list(filter(str.isalpha, alts))
    alts = alts[0] if len(alts) == 1 else alts
    if try_convert:
        if chrom != 'X' and chrom == 'Y':
            chrom = int(chrom)
        pos = int(pos)
    return chrom, pos, ref, alts


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


def get_all_files_extension(folder, extension):
    import os
    all_files = []
    for root, _, filenames in os.walk(folder):
        for filename in filenames: 
            all_files.append(os.path.join(root,filename))

    all_wanted = [f for f in all_files if f.endswith(extension)]
    return all_wanted


def get_dtypes_info(dtype):
    if "float" in str(dtype):
        return (np.finfo(dtype).min, np.finfo(dtype).max)
    else:
        return (np.iinfo(dtype).min, np.iinfo(dtype).max)


if __name__ == '__main__':
    inputfile = get_data_dir() + "/raw/v1.3/training_data/training_data.tsv"
    shuffled_idxs_file = get_data_dir() + "/raw/v1.3/training_data/shuffle_splits/shuffled_index.pickle"
    variant_id_file = get_data_dir() + "/raw/v1.3/training_data/variant_ids.pkl"
    outputfile = get_data_dir() + "/raw/v1.3/training_data/shuffle_splits/batch_idxs_256.pkl"
    generate_batch_idxs(shuffled_idxs_file, variant_id_file, 256, outputfile)

