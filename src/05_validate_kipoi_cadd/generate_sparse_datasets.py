from kipoi_cadd.utils import dump_to_pickle, load_pickle, get_all_files_extension, generate_variant_ids
from kipoi_cadd.data_utils import load_csv_chunks_tosparse
from kipoi_cadd.data import sparse_cadd_dataset, CaddSparseDataset
from scipy.sparse import vstack, load_npz, save_npz
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm
import pandas as pd
import argparse
import numpy as np
import os

VALIDATION_DATA_FILES = [
    "sample_chr22_GRCh37"
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Input directory.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("--split", default=False, action='store_true')
    parser.add_argument("--scaler-path", type=str, default=None)
    args = parser.parse_args()

    print("Got args:")
    for arg in vars(args):
        print("\t" + arg + ":", getattr(args, arg))

    training_dir = args.input_dir
    output_dir = args.output_dir
    variant_ids_dir = os.path.join(output_dir, "variant_ids")
    sparse_matrices_dir = os.path.join(output_dir, "sparse_matrices")

    if not os.path.exists(variant_ids_dir):
        os.makedirs(variant_ids_dir)
    if not os.path.exists(sparse_matrices_dir):
        os.makedirs(sparse_matrices_dir)

    variant_cols = ['Chrom', 'Pos', 'Ref', 'Alt']
    dtype = {'Chrom': 'str', 'Pos': np.int32, 'Ref': 'str',
                'Alt': 'str'}

    print("Extracting variant ids...")
    # Extract the variant ids
    for f in VALIDATION_DATA_FILES:
        inputfile = os.path.join(training_dir, f + ".tsv.gz")
        out = os.path.join(variant_ids_dir, f + ".pkl")
        if not os.path.isfile(out): # skip if file exists
            generate_variant_ids(inputfile, out, header=None, variant_cols=variant_cols,
                                 dtype=dtype, comment="#")

    print("Generating sparse matrices...")
    # Generate sparse matrices
    for f in VALIDATION_DATA_FILES:
        # Get the base of the name
        inputfile = os.path.join(training_dir, f + ".csv.gz")
        # Num lines is necessary to set the total in tqdm, important feedback in a lengthy function
        num_lines = len(load_pickle(os.path.join(variant_ids_dir, f + ".pkl")))
        output = os.path.join(sparse_matrices_dir, f + ".npz")
        if not os.path.isfile(output):
            load_csv_chunks_tosparse(inputfile, 10000, np.float32, num_lines=num_lines,
                                     output=output, header=None)

    # Merge variant ids
    output = os.path.join(variant_ids_dir, "all.pkl")
    if not os.path.isfile(output):
        print("Merging variant ids...")
        all_ids = None
        for f in tqdm(VALIDATION_DATA_FILES):
            inputfile = os.path.join(variant_ids_dir, f + ".pkl")
            if all_ids is None:
                all_ids = load_pickle(inputfile)
            else:
                all_ids = pd.concat([all_ids, load_pickle(inputfile)], ignore_index=True)
        dump_to_pickle(output, all_ids)

    # Merge sparse matrices
    output = os.path.join(sparse_matrices_dir, "all.npz")
    if not os.path.isfile(output):
        print("Merging sparse matrices...")
        all_npz = None
        for f in tqdm(VALIDATION_DATA_FILES):
            inputfile = os.path.join(sparse_matrices_dir, f + ".npz")
            if all_npz is None:
                all_npz = load_npz(inputfile)
            else:
                all_npz = vstack([all_npz, load_npz(inputfile)])
        save_npz(output, all_npz)

    # Split into train and test sets
    if args.split:
        print("Splitting into train and test sets...")
        variant_ids_file = os.path.join(variant_ids_dir, "all.pkl")
        s = os.path.join(sparse_matrices_dir, "all.npz")
        (train, train_ids), (valid, valid_ids) = sparse_cadd_dataset(
            s, variant_ids_file, output_npz=sparse_matrices_dir,
            output_ids=variant_ids_dir)

    # Creating scaler (will base itself on the train set if data was split)
    if args.scaler_path is not None:
        print("Fitting a scaler...")
        from sklearn.preprocessing import StandardScaler
        from sklearn.externals import joblib

        trainfile = os.path.join(sparse_matrices_dir, "train.npz")
        allfile = os.path.join(sparse_matrices_dir, "all.npz")
        if os.path.isfile(trainfile):
            inputfile = trainfile
            idsfile = os.path.join(variant_ids_dir, "train.pkl")
        elif os.path.isfile(allfile):
            inputfile = allfile
            idsfile = os.path.join(variant_ids_dir, "all.pkl")
        else:
            raise FileNotFoundError(trainfile + " and " + allfile + " not found.")
        
        ds = CaddSparseDataset(inputfile, idsfile, "v1.4")
        scaler_kipoicadd = StandardScaler(copy=True, with_mean=False, with_std=True)
        X, y = ds.load_all()
        scaler_kipoicadd.fit(X)
        joblib.dump(scaler_kipoicadd, args.scaler_path)