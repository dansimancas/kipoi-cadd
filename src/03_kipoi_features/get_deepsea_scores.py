# Standalone script
import kipoi_veff.snv_predict as sp
from kipoi_veff.scores import Logit
from pathlib import Path
from kipoi import get_model
import argparse
import os
import pyarrow as pa
from kipoi_veff.utils.io import SyncPredictonsWriter, SyncBatchWriter
import pandas as pd
from tqdm import tqdm
import numpy as np
import logging
import lmdb
import pickle


def load_pickle(filename):
    with open(filename, 'rb') as f:
        rn = pickle.load(f)
    return rn

def calculate_map_size(row_example, nrows, multiplier=1.9):
    row_size = pa.serialize(row_example).to_buffer().size
    map_size = int(row_size * nrows * multiplier)
    return map_size

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


def concatenate_vcf_files(directory, filenames, output=None):
    ext = "vcf.gz"
    vcf = None
    col_names = ['#CHROM', 'POS', 'ID', 'REF', 'ALT']
    for fil in filenames:
        f = os.path.join(directory, fil)
        if vcf is None:
            vcf = pd.read_csv(f, sep='\t', comment='#', names=col_names,
                              dtype={0:'str',
                                      1:'int32',
                                      2:'str',
                                      3:'str',
                                      4:'str'})
        else:
            vcf = pd.concat([vcf, 
                             pd.read_csv(f, sep='\t', comment='#', names=col_names,
                                         dtype={0:'str',
                                                 1:'int32',
                                                 2:'str',
                                                 3:'str',
                                                 4:'str'})], ignore_index=True)
        print(f)
    vcf.sort_values(by=['#CHROM', 'POS'], inplace=True)
    vcf.reset_index(drop=True, inplace=True)
    
    vcf["QUAL"] = ['.'] * vcf.shape[0]
    vcf["FILTER"] = ['.'] * vcf.shape[0]
    vcf["INFO"] = ['.'] * vcf.shape[0]
    
    if output is not None:
        with open(output, 'w') as f:
            f.write("##fileformat=VCFv4.0\n")
        vcf.to_csv(output, sep='\t', index=None, mode='a')
    return vcf

def _write_worker(q, sync_pred_writer):
    """Writer loop
    Args:
      q: multiprocessing.Queue
      batch_writer.
    """
    while True:
        batch = q.get()
        if batch is None:
            return
        else:
            sync_pred_writer(*batch)

class AsyncSyncPredictionsWriter(SyncPredictonsWriter):

    def __init__(self, sync_pred_writer, max_queue_size=100):
        """
        Args:
          batch_writer: BatchWriter object
          max_queue_size: maximal queue size. If it gets
            larger then batch_write needs to wait
             till it can write to the queue again.
        """
        from multiprocessing import Queue, Process
        self.sync_pred_writer = sync_pred_writer
        self.max_queue_size = max_queue_size

        # instantiate the queue and start the process
        self.queue = Queue()
        self.process = Process(target=_write_worker,
                               args=(self.queue, self.sync_pred_writer))
        self.process.start()

        
    def __call__(self, predictions, records, line_ids=None):
        """Write a single batch of data
        Args:
          batch is one batch of data (nested numpy arrays with the same axis 0 shape)
        """
        import time
        batch = (predictions, records, line_ids)
        while self.queue.qsize() > self.max_queue_size:
            print("WARNING: queue too large {} > {}. Blocking the writes".
                  format(self.queue.qsize(), self.max_queue_size))
            time.sleep(1)
        self.queue.put(batch)

    def batch_write(self, batch):
        predictions, line_ids = batch['preds'], batch['line_idx']
        rec = {k: batch[k] for k in batch.keys() if k != 'preds' and k != 'line_idx'}
        self.__call__(predictions, rec, line_ids)

    def close(self):
        """Close the file
        """
        # stop the process,
        # make sure the queue is empty
        # close the file
        self.process.join(.5)  # wait one second to close it
        if self.queue.qsize() > 0:
            print("WARNING: queue not terminated successfully. {} elements left".
                  format(self.queue.qsize()))
            print(self.queue.get())
        self.sync_pred_writer.close()
        self.process.terminate()

    def __del__(self):
        self.close()

class LmdbBatchWriter(SyncPredictonsWriter):
    """Synchronous writer for output on an LMDB database. Uses each variant id as key,
    pointing to the predictions on that record.
    
    # Arguments:
      lmdb_dir: str - Directory where the lmdb database will be created.
      map_size: int - Map size used to initialize the lmdb (number of bytes).
    """
    def __init__(self, lmdb_dir, db_name, map_size=10E8):
        self.lmdb_dir = lmdb_dir
        self.db_name = db_name
        self.map_size = map_size
    
    def __call__(self, predictions, records, line_ids=None, merge_preds=False):
        import pyarrow as pa
        import cyvcf2
        
        if merge_preds:
            merged_preds = None
            for k in predictions.keys():
                if isinstance(predictions[k], pd.DataFrame):
                    coldict = {c: self.db_name + ":" + k + "_" + c for c in predictions[k].columns.values}
                    predictions[k].rename(columns=coldict, inplace=True)
                    if merged_preds is None:
                        merged_preds = predictions[k]
                    else:
                        merged_preds = merged_preds.join(predictions[k], how='outer')
        
            self.env = lmdb.open(self.lmdb_dir, map_size=self.map_size, max_dbs=0, lock=False)
            with self.env.begin(write=True) as txn:
                for var_num, var in enumerate(records):
                    variant_id = variant_id_string(
                        records['variant_chr'][var_num], 
                        records['variant_pos'][var_num],
                        records['variant_ref'][var_num],
                        records['variant_alt'][var_num])
                    # Obtain predictions for this variant...
                    annos = merged_preds[var_num]

                    buf = pa.serialize(annos).to_buffer()
                    txn.put(variant_id.encode('ascii'), buf)
        else:

            self.env = lmdb.open(self.lmdb_dir, map_size=self.map_size, max_dbs=0, lock=False)
            with self.env.begin(write=True) as txn:
                for var_num, var in enumerate(records):
                    variant_id = variant_id = variant_id_string(
                        records['variant_chr'][var_num], 
                        records['variant_pos'][var_num],
                        records['variant_ref'][var_num],
                        records['variant_alt'][var_num])
                    annos = {}
                    # Obtain predictions for this variant...
                    
                    for key, preds in predictions.items():
                        annos[key] = {}
                        for k, arr in preds.items():
                            annos[key][k] = arr[var_num]

                    buf = pa.serialize(annos).to_buffer()
                    txn.put(variant_id.encode('ascii'), buf)

                
    def batch_write(self, batch):
        predictions, records, line_ids = batch
        self.__call__(predictions, records, line_ids)
        
        
    def batch_write(self, predictions, records, line_ids):
        self.__call__(predictions, records, line_ids)
    
    
    def close(self):
        if hasattr(self, 'env'):
            self.env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('vcf')
    parser.add_argument('--output-dir', default="/s/project/kipoi-cadd/data/models/DeepSea_veff")
    parser.add_argument("--writer", default="zarr")
    args = parser.parse_args()
    
    fasta_file = "/s/genomes/human/hg19/ensembl_GRCh37.p13_release75/Homo_sapiens.GRCh37.75.dna.primary_assembly.fa"

    model = get_model("DeepSEA/variantEffects")
    dl_kwargs = {'fasta_file': fasta_file, 'num_chr_fasta': True}
    output_dir = Path(args.output_dir)
    output_name = os.path.basename(args.vcf).split('.vcf')[0]

    if args.writer == "zarr":
        from kipoi.writers import ZarrBatchWriter, AsyncBatchWriter
        td = output_name + ".zarr"
        writer = SyncBatchWriter(AsyncBatchWriter(ZarrBatchWriter(str(output_dir / td), chunk_size=1024)))
    elif args.writer == "lmdb":
        td = output_name + ".lmdb"
        writer = SyncBatchWriter(AsyncSyncPredictionsWriter(
            LmdbBatchWriter(str(output_dir / td), "DeepSea_veff", 274578419865)))
    elif args.writer == "hdf5":
        td = output_name + ".hdf5"
        from kipoi.writers import HDF5BatchWriter
        writer = SyncBatchWriter(HDF5BatchWriter(str(output_dir / td)))
    
    print("Start predictions..")
    sp.score_variants(model=model,
                input_vcf=args.vcf,
                batch_size=16,
                num_workers=10,
                dl_args=dl_kwargs,
                output_writers=writer)
    
