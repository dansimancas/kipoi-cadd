"""Test lmdb functionality
"""
import lmdb
import pandas as pd
import pyarrow as pa
import numpy as np
from kipoi_cadd.config import get_data_dir


def test_lmdb_get_put():
    ddir = get_data_dir()
    X = pd.read_csv(ddir + "/raw/v1.3/training_data/shuffle_splits/testing/3471.csv", index_col=0)
    Xnp = np.array(X)
    env = lmdb.Environment(ddir + "/tests/lmdb_1.mdb", map_size=Xnp.nbytes*20, max_dbs=0, lock=False)
    check_index = X.index[10]

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
    assert variant['inputs'].equals(s[1:])


def test_pyarrow_serialization():
    pass


test_lmdb_get_put()