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
