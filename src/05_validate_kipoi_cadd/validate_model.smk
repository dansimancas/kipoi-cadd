import os
import subprocess

CADD_DIR = "/s/project/kipoi-cadd/data/CADD-scripts"
TMP_DIR = os.path.join(CADD_DIR, "/data/tmp")

rule annotate_vcf:
    input:
        infile = "/s/project/kipoi-cadd/data/raw/v1.4/validation/clinVar-ExAC/clinvar_20180729_pathogenic_all_{genomebuild}.vcf",
    output:
        outfile = "/s/project/kipoi-cadd/data/raw/v1.4/validation/clinVar-ExAC/clinvar_20180729_pathogenic_all_{genomebuild}.tsv"
    run:
        fileformat = {wildcards.format}
        filename = os.path.basename({input.infile}).split(fileformat)[0][:-1]
        print("Filename:", filename)
        if not (infile.endswith("vcf") or infile.endswith("vcf.gz")):
            raise ValueError("Unknown file format {fileformat}. Make sure you provide a *.vcf or *.vcf.gz file.")
        if {wildcards.genomebuild} != "GRCh38" and {wildcards.genomebuild} != "GRCh37":
            raise ValueError("Unknown/Unsupported genome build {wildcards.genomebuild}. CADD currently only supports GRCh37 and GRCh38.")
        
        # Pipeline configuration
        PRESCORED_FOLDER = "{CADD_DIR}/data/prescored/{wildcards.genomebuild}/incl_anno"
        REFERENCE_CONFIG = "{CADD_DIR}/config/{wildcards.genomebuild}references.cfg"
        IMPUTE_CONFIG = "{CADD_DIR}/config/impute_{wildcards.genomebuild}.cfg"
        MODEL = "{CADD_DIR}/data/models/{wildcards.genomebuild}/CADD1.4-{wildcards.genomebuild}.mod"
        CONVERSION_TABLE = "{CADD_DIR}/data/models/{wildcards.genomebuild}/conversionTable_CADD1.4-{wildcards.genomebuild}.txt"

        # Temp files
        TMP_FOLDER = "{CADD_DIR}/data/tmp"
        TMP_PRE = "{TMP_FOLDER}/{filename}.pre.tsv.gz"
        TMP_VCF = "{TMP_FOLDER}/{filename}.vcf"
        TMP_ANNO = "{TMP_FOLDER}/{filename}.anno.tsv.gz"
        TMP_IMP = "{TMP_FOLDER}/{filename}.csv.gz"
        TMP_NOV = "{TMP_FOLDER}/{filename}.nov.tsv.gz"

        if fileformat == "vcf":
            cmd = "cat {input.infile} | python {CADD_DIR}/src/scripts/VCF2vepVCF.py | sort -k1,1 -k2,2n -k3,3 -k4,4 \
            | uniq > {TMP_VCF}"
            subprocess.check_output(cmd, shell=True)
        else:
            cmd = "zcat {input.infile} | python {CADD_DIR}/src/scripts/VCF2vepVCF.py | sort -k1,1 -k2,2n -k3,3 -k4,4 \
            | uniq > {TMP_VCF}"
            subprocess.check_output(cmd, shell=True)

        # Variant annotation
        cmd = "cat {TMP_VCF} | vep --quiet --cache --buffer 1000 --no_stats --offline --vcf --dir \
        {CADD_DIR}/data/annotations/{wildcards.genomebuild}/vep --species homo_sapiens --db_version=92 \
        --assembly {wildcards.genomebuild} --regulatory --sift b --polyphen b --per_gene --ccds --domains\
        --numbers --canonical --total_length --force_overwrite --format vcf --output_file STDOUT \
        --warning_file STDERR | python {CADD_DIR}/src/scripts/annotateVEPvcf.py -c {REFERENCE_CONFIG} \
        | gzip -c > {TMP_ANNO}"
        subprocess.check_output(cmd, shell=True)
        subprocess.run(["rm", "{TMP_VCF}"])

rule inputate_annos:
    input:



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
            if [ "{wildcards.genomebuild}" == "GRCh38" ]
            then
                COLUMNS="1-4,124,125"
            else
                COLUMNS="1-4,106,107"
            fi
            zcat $TMP_NOV | cut -f $COLUMNS | uniq | gzip -c > $TMP_NOV.tmp
            mv $TMP_NOV.tmp $TMP_NOV
        fi
        """
