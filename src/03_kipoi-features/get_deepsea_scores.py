# Standalone script
import kipoi_veff.snv_predict as sp
from kipoi_veff.scores import Logit
from kipoi import get_model
import os
import pyarrow as pa
from kipoi_veff.utils.io import SyncPredictonsWriter
import pandas as pd
from tqdm import tqdm
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

class LmdbWriter(SyncPredictonsWriter):
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
    
    def __call__(self, predictions, records, line_ids=None):
        import pyarrow as pa
        
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
            for var_num, var in tqdm(enumerate(records), total=len(records)):
                variant_id = variant_id_string(var.CHROM, var.POS, var.REF, var.ALT)
                # Obtain predictions for this variant...
                annos = merged_preds.iloc[var_num, :]

                buf = pa.serialize(annos).to_buffer()
                txn.put(variant_id.encode('ascii'), buf)
  
    def close(self):
        if self.env is not None:
            self.env.close()

cadd_files_dir = "/data/ouga/home/ag_gagneur/simancas/Projects/kipoi-veff/tests/models/var_seqlen_model/"
training_dir_hg37 = "/s/project/kipoi-cadd/data/raw/v1.4/training_data/GRCh37"
intervals_file = os.path.join(training_dir_hg37, "intervals.tsv")
fasta_file = "/s/genomes/human/hg19/ensembl_GRCh37.p13_release75/Homo_sapiens.GRCh37.75.dna.primary_assembly.fa"
vcf_file = os.path.join(training_dir_hg37, "all.vcf.gz")
lmdb_deep_sea = os.path.join(training_dir_hg37, "lmdb/lmdb_DeepSea_veff")

model = get_model("DeepSEA/variantEffects")
dl_kwargs = {'intervals_file': intervals_file, 'fasta_file': fasta_file, 'num_chr_fasta': True}
dataloader = model.default_dataloader

# num_lines = len(load_pickle(os.path.join(training_dir_hg37, "variant_ids/all.pkl")))
# map_size = calculate_map_size(ds[0], num_lines, 1.9)
writer = LmdbWriter(lmdb_deep_sea, "DeepSea_veff", 274578419865)

if __name__ == "__main__":
    sp.predict_snvs(model,
                dataloader,
                vcf_file,
                16,
                num_workers=64,
                dataloader_args=dl_kwargs,
                evaluation_function_kwargs={'diff_types': {'logit': Logit()}},
                return_predictions=False,
                sync_pred_writer=writer)