import argh
from kipoi_cadd.cli.train import train_keras
from kipoi_cadd.data import create_lmdb
from kipoi_cadd.tests.test_lmdb import test_update_targets, test_put_10000_variants

def wc_l(fname):
    """Get the number of lines of a text-file using unix `wc -l`
    """
    import os
    import subprocess
    return int((subprocess.Popen('wc -l {0}'.format(fname), shell=True, stdout=subprocess.PIPE).stdout).readlines()[0].split()[0])


def main():
    # assembling:
    parser = argh.ArghParser()
    parser.add_commands([wc_l, train_keras, create_lmdb, test_update_targets, test_put_10000_variants])
    argh.dispatch(parser)
