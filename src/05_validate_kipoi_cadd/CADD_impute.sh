#!/bin/bash

while getopts ':c:g:i:o:' option; do
  case "$option" in
    c) CADD=$OPTARG
       ;;
    g) GENOMEBUILD=$OPTARG
       ;;
    i) INFILE=$OPTARG
       ;;
    o) OUTFILE=$OPTARG
       ;;
    \?) echo "Invalid option -$OPTARG" >&2
       ;;
  esac
done

FILENAME=$(basename $INFILE)
NAME=${FILENAME%\.anno*}
TMP_FOLDER=$CADD/data/tmp
TMP_ANNO=$TMP_FOLDER/$NAME.anno.tsv.gz
IMPUTE_CONFIG=$CADD/config/impute_$GENOMEBUILD.cfg

# Loading the environment
source activate cadd-env

# Imputation
zcat $TMP_ANNO \
| python $CADD/src/scripts/trackTransformation.py -b \
            -c $IMPUTE_CONFIG -o $OUTFILE --noheader;