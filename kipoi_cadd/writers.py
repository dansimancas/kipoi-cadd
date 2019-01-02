from kipoi_veff.utils.io import SyncPredictonsWriter
from kipoi_cadd.utils import variant_id_string
import logging
import lmdb

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class LmdbWriter(SyncPredictonsWriter):
    """Synchronous writer for output on an LMDB database. Uses each variant id as key,
    pointing to the predictions on that record.
    
    # Arguments:
      lmdb_dir: str - Directory where the lmdb database will be created.
      map_size: int - Map size used to initialize the lmdb (number of bytes).
    """
    def __init__(self, lmdb_dir, map_size=10E8):
        self.lmdb_dir = lmdb_dir
        self.map_size = map_size
    
    def __call__(self, predictions, records, line_ids=None):
        import pyarrow as pa
        
        self.env = lmdb.open(self.lmdb_dir , map_size=self.map_size, max_dbs=0, lock=False)
        with self.env.begin(write=True) as txn:
            for var_num, var in enumerate(records):
                variant_id = variant_id_string(var.CHROM, var.POS, var.REF, var.ALT)
                annotations = {}
                for p in predictions:
                    # Verify there is a prediction for this variant...
                    annotations[p] = predictions[p].iloc[var_num, :]

                buf = pa.serialize(annotations).to_buffer()
                txn.put(variant_id.encode('ascii'), buf)
  
    def close(self):
        if self.env is not None:
            self.env.close()