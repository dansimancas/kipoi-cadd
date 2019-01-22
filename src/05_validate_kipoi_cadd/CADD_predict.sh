#!/bin/bash

while getopts ':c:g:i:o:k:' option; do
  case "$option" in
    c) CADD=$OPTARG
       ;;
    g) GENOMEBUILD=$OPTARG
       ;;
    i) INFILE=$OPTARG
       ;;
    o) OUTFILE=$OPTARG
       ;;
    k) KEEP=$OPTARG
       ;;
    \?) echo "Invalid option -$OPTARG" >&2
       ;;
  esac
done

ANNOTATION="true"
MODEL=$CADD/data/models/$GENOMEBUILD/CADD1.4-$GENOMEBUILD.mod
CONVERSION_TABLE=$CADD/data/models/$GENOMEBUILD/conversionTable_CADD1.4-$GENOMEBUILD.txt
FILENAME=$(basename $INFILE)
NAME=${FILENAME%\.csv*}
TMP_FOLDER=$CADD/data/tmp
TMP_PRE=$TMP_FOLDER/$NAME.pre.tsv.gz
TMP_ANNO=$TMP_FOLDER/$NAME.anno.tsv.gz
TMP_NOV=$TMP_FOLDER/$NAME.nov.tsv.gz

# Loading the environment
source activate cadd-env

# Score prediction
python $CADD/src/scripts/predictSKmodel.py \
    -i $INFILE -m $MODEL -a $TMP_ANNO \
| python $CADD/src/scripts/max_line_hierarchy.py --all \
| python $CADD/src/scripts/appendPHREDscore.py -t $CONVERSION_TABLE \
| gzip -c > $TMP_NOV;
rm $TMP_ANNO

if [ "$KEEP" = 'false' ]
then
    rm $INFILE
fi

if [ "$ANNOTATION" = 'false' ]
then
    if [ "$GENOMEBUILD" == "GRCh38" ]
    then
        COLUMNS="1-4,124,125"
    else
        COLUMNS="1-4,106,107"
    fi
    zcat $TMP_NOV | cut -f $COLUMNS | uniq | gzip -c > $TMP_NOV.tmp
    mv $TMP_NOV.tmp $TMP_NOV
fi

# Join pre and novel scored variants
{
    echo "##CADD $GENOMEBUILD-v1.4 (c) University of Washington, Hudson-Alpha Institute for Biotechnology and Berlin Institute of Health 2013-2018. All rights reserved.";
    head -n 1 < <(zcat $TMP_NOV);
    zcat $TMP_PRE $TMP_NOV | grep -v "^#" | sort -k1,1 -k2,2n -k3,3 -k4,4 || true;
} | bgzip -c > $OUTFILE;
rm $TMP_NOV
rm $TMP_PRE

OUTFILE=$(echo $OUTFILE |  sed 's/^\.\///')
echo -e "\nCADD scored variants written to file: $OUTFILE"

exit 0