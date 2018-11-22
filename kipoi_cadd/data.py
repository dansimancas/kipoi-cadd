"""Functions to load the data
"""
import dask.dataframe as dd
from kipoi.data import BatchDataset
import pyarrow as pa
import lmdb


class CaddDataset(BatchDataset):
    def __init__(self, version="1.3", exclude_idx=None, include_idx=None):
        self.version = version

        # indexed by location
        self.df_index = load_rownames()

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

        return pa.deserialize(self.lmdb_cadd.get(variant_id))


def cadd_training_set():
    # return (train, valid) datasets
    return CaddDataset(exclude=valid_idx), CaddDataset(include=valid_idx)


def create_lmdb(df, output_lmdb_file):
    db = lmdb.Environment(output_lmdb_file)

    for i in range(len(X)):
        data = {"inputs": df[i, feature_columns],
                "targets": df[i, target_columns],
                "metadata": {"variant_id": df.index[i],
                             "row_idx": i}}

        buf = pa.serialize(data).to_buffer()
        # db.put(key, value)
        db.put(data['metadata']['variant_id']['variant_id'], buf)


# TODO - make sure you can serialize and de-serialize using pyarraow
# make sure you can run put and get


def cadd_training(version='1.3'):
    return dd.read_csv(f'/s/project/kipoi-cadd/data/v{version}/training_data.tsv.gz')
