import collections
import sys
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
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

    
def generate_intervals_from_vcf(vcf,
                                output=None,
                                col_names=['#CHROM', 'POS', 'ID', 'REF', 'ALT'],
                                dtypes={'#CHROM': 'str', 'POS': 'int32', 'ID': 'str', 'REF':'str', 'ALT':'str'}):
    if isinstance(vcf, str):
        vcf = pd.read_csv(vcf,
                          sep='\t',
                          dtype=dtypes,
                          header= None,
                          names=col_names,
                          usecols=range(len(col_names)),
                          comment='#')
    elif not isinstance(sparse_matrix, pd.DataFrame):
        raise ValueError("Input must be either a path to a vcf(.gz) file or an object of pd.DataFrame type.")
    
    intervals = {'chr': [], 'start': [], 'end': []}
    for _, row in tqdm(vcf.iterrows(), total=vcf.shape[0]):
        intervals['chr'].append(row['#CHROM'])
        intervals['start'].append(row['POS'] - 1)
        intervals['end'].append((row['POS'] - 1) + len(row['REF']))
    
    df = pd.DataFrame(intervals, index=range(len(intervals['chr'])))
    df.sort_values(by=['chr', 'start'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    if output is not None:
        df.to_csv(output, sep='\t', index=None, header=None)
    return df


def concatenate_vcf_files(directory, output=None):
    ext = "vcf.gz"
    vcf = None
    col_names = ['#CHROM', 'POS', 'ID', 'REF', 'ALT']
    for f in get_all_files_extension(training_dir_hg37, ext):
        if vcf is None:
            vcf = pd.read_csv(f, sep='\t', comment='#', names=col_names,
                              dtypes={0:'str',
                                      1:'int32',
                                      2:'str',
                                      3:'str',
                                      4:'str'})
        else:
            vcf = pd.concat([vcf, 
                             pd.read_csv(f, sep='\t', comment='#', names=col_names,
                                         dtypes={0:'str',
                                                 1:'int32',
                                                 2:'str',
                                                 3:'str',
                                                 4:'str'})], ignore_index=True)
        print(f)
    # vcf.astype(dtype={'#CHROM':'object', 'POS':'int32', 'ID':'object', 'REF':'object', 'ALT':'object'})
    vcf.sort_values(by=['#CHROM', 'POS'], inplace=True)
    vcf.reset_index(drop=True, inplace=True)
    if output is not None:
        vcf.to_csv(os.path.join(training_dir_hg37, "all.vcf.gz"), sep='\t', index=None)
    return vcf


def generate_variant_ids(inputfile, outputfile, separator='\t',
                         header=0,
                         variant_cols=['Chrom', 'Pos', 'Ref', 'Alt'],
                         dtype={'Chrom': 'str', 'Pos': np.int32, 'Ref': 'str',
                                'Alt': 'str'}):
    
    print(inputfile)
    with open(inputfile, 'r') as f:
        for l in f.readlines():
            print(l)
            break

    input_df = pd.read_csv(inputfile,
                           sep=separator,
                           header=header,
                           usecols=variant_cols,
                           # nrows=1000,
                           dtype=dtype)
    
    if header is None:
        # Make sure column numbers are reset
        input_df = input_df.T.reset_index(drop=True).T
    variant_ids = input_df.apply(
        lambda row: ':'.join([str(row[0]), str(row[1]), row[2],
                              str(row[3].split(','))]), axis=1)
    
    print(outputfile)
    dump_to_pickle(outputfile, variant_ids)


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


def get_all_files_extension(folder, extension, recursive=False, full_path=True):
    import os
    
    all_files = []
    if recursive:
        for root, _, filenames in os.walk(folder):
            for filename in filenames:
                if full_path:
                    all_files.append(os.path.join(root,filename))
                else:
                    all_files.append(filename)
    else:
        tmp = [f for f in os.listdir(folder) if os.path.isfile(
            os.path.join(folder, f))]
        for filename in tmp:
            if full_path:
                all_files.append(os.path.join(folder,filename))
            else:
                all_files.append(filename)
    all_wanted = [f for f in all_files if f.endswith(extension)]

    return all_wanted


def get_dtypes_info(dtype):
    if "float" in str(dtype):
        return (np.finfo(dtype).min, np.finfo(dtype).max)
    else:
        return (np.iinfo(dtype).min, np.iinfo(dtype).max)

