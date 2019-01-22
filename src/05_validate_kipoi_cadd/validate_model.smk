filename_root = "clinvar_20180729_pathogenic_all_"


rule annotate_vcf:
    input:
        # "{data_dir}/raw/v1.4/training_data/{genomebuild}/" + filename_root + "{genomebuild}.vcf"
        "{data_dir}/raw/v1.4/validation/clinVar-ExAC/" + filename_root + "{genomebuild}.vcf"
    output:
        "{data_dir}/CADD-scripts/data/tmp/" + filename_root + "{genomebuild}.anno.tsv.gz"
    params:
        cadd_dir="{data_dir}/CADD-scripts"
    shell:
        """
        bash ./CADD_annotate.sh -c {params.cadd_dir} -i {input} -g {wildcards.genomebuild} -a
        """

rule impute_annotations:
    input:
        "{data_dir}/CADD-scripts/data/tmp/" + filename_root + "{genomebuild}.anno.tsv.gz"
    output:
        # "{data_dir}/CADD-scripts/data/tmp/" + filename_root + "{genomebuild}.csv.gz"
        "{data_dir}/raw/v1.4/validation/clinVar-ExAC/" + filename_root + "{genomebuild}.csv.gz"
    params:
        cadd_dir="{data_dir}/CADD-scripts"
    shell:
        """
        bash ./CADD_impute.sh -c {params.cadd_dir} -g {wildcards.genomebuild} -i {input} -o {output}
        """

rule predict_scores:
    input:
        "{data_dir}/raw/v1.4/validation/clinVar-ExAC/" + filename_root + "{genomebuild}.csv.gz"
        # "{data_dir}/CADD-scripts/data/tmp/" + filename_root + "{genomebuild}.csv.gz"
    output:
        # "{data_dir}/raw/v1.4/training_data/{genomebuild}/" + filename_root + "{genomebuild}.tsv.gz"
        "{data_dir}/raw/v1.4/validation/clinVar-ExAC/" + filename_root + "{genomebuild}.tsv.gz"
    params:
        cadd_dir="{data_dir}/CADD-scripts"
    shell:
        """
        bash ./CADD_predict.sh -c {params.cadd_dir} -g {wildcards.genomebuild} -i {input} -o {output} -k true
        """