# Score prediction
python $CADD/src/scripts/predictSKmodel.py \
    -i $TMP_IMP -m $MODEL -a $TMP_ANNO \
| python $CADD/src/scripts/max_line_hierarchy.py --all \
| python $CADD/src/scripts/appendPHREDscore.py -t $CONVERSION_TABLE \
| gzip -c > $TMP_NOV;
rm $TMP_ANNO
rm $TMP_IMP

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