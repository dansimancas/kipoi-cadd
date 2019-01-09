from kipoi.readers import Reader
import lmdb

class LmdbReader(Reader):
    def __init__(self, lmdb_dir):
        self.lmdb_dir = lmdb_dir
        self.env = lmdb.open(self.lmdb_dir, readonly=True, lock=False)
        self.txn = self.env.begin()
        
        
    def __len__(self):
        length = self.txn.stat()['entries']
        return length
    
    def __del__(self):
        if self.env is not None:
            self.env.close()
    
    def single_iter(self):
        """Iterator returns a tuple (key, value) when calling `next` on it.
        `key` and `value` are encoded as they were when written into LMDB.
        """
        return iter(self.txn.cursor())
        
    close = __del__