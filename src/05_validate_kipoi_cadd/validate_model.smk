import os

CADD_DIR = "/s/project/kipoi-cadd/data/CADD-scripts"
TMP_DIR = os.path.join(CADD_DIR, "/data/tmp")
TMP_ANNO = os.path.join(TMP_DIR, "clinvar_20180729_pathogenic_all_GRCh37.anno.tsv.gz")
TMP_IMP = os.path.join(TMP_DIR, "clinvar_20180729_pathogenic_all_GRCh37.csv.gz")
GENOMEBUILD = "GRCh37"
MODEL = os.path.join(CADD_DIR, "/data/models/", GENOMEBUILD, "/CADD1.4-", GENOMEBUILD, ".mod")

rule annotate_vcf:
    input:
        infile = "/s/project/kipoi-cadd/data/raw/v1.4/validation/clinVar-ExAC/clinvar_20180729_pathogenic_all_GRCh37.vcf",
    output:
        outfile = "/s/project/kipoi-cadd/data/raw/v1.4/validation/clinVar-ExAC/clinvar_20180729_pathogenic_all_GRCh37.tsv"
    shell:
        "bash CADD_annotate.sh -c {CADD_DIR} -o {output.outfile} -g {GENOMEBUILD} -a {input.infile}"


rule predict:
    input:
        infile = "/s/project/kipoi-cadd/data/raw/v1.4/validation/clinVar-ExAC/clinvar_20180729_pathogenic_all_GRCh37.tsv"
    output:
        outfile = "bla"
    shell:
        """
        # Score prediction
        python {CADD_DIR}/src/scripts/predictSKmodel.py \
            -i {TMP_IMP} -m {MODEL} -a {TMP_ANNO} \
        | python {CADD_DIR}/src/scripts/max_line_hierarchy.py --all \
        | python {CADD_DIR}/src/scripts/appendPHREDscore.py -t $CONVERSION_TABLE \
        | gzip -c > $TMP_NOV;
        rm {TMP_ANNO}
        rm {TMP_IMP}
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
        """
