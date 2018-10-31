"""Generic kipoi rule for annotating the vcf files
"""
from m_kipoi.utils import get_env_executable

# ENV_NAME = "kipoi-splicing"
ENV_NAME = "kipoi-DeepBind__Homo_sapiens__RBP__D00084.001_RNAcompete_A1CF"
GTF_FILE = "data/raw/dataloader_files/Homo_sapiens.GRCh37.75.filtered.gtf"
FASTA_FILE = "data/raw/dataloader_files/hg19.fa"
# MODELS_FULL = "KipoiSplice/4cons"
MODELS_FULL = "DeepBind/Homo_sapiens/RBP/D00084.001_RNAcompete_A1CF"

rule create__env:
    """Create one conda environment for all splicing models
    """
    output:
        env = get_env_executable(ENV_NAME)
    shell:
        "kipoi env create {MODELS_FULL} -e {ENV_NAME} --vep"


rule annotate_vcf:
    """Annotate the Vcf using Kipoi's score variants
    """
    input:
        vcf = "data/processed/kipoi/chr{d}/{vcf_file}.vcf.gz",
        gtf = "data/raw/dataloader_files/Homo_sapiens.GRCh37.75.filtered.gtf",
        fasta = "data/raw/dataloader_files/hg19.fa",
        kipoi = get_env_executable(ENV_NAME)
    output:
        h5 = "data/processed/splicing/chr{d}/annotated_vcf/{vcf_file}/models/{model}.h5"
    params:
        dl_kwargs = json.dumps({"gtf_file": os.path.abspath(GTF_FILE),
                                "fasta_file": os.path.abspath(FASTA_FILE)}),
    shell:
        """
        mkdir -p `dirname {output.h5}`
        {input.kipoi} predict \
            {wildcards.model} \
            --dataloader_args='{params.dl_kwargs}' \
            -i {input.vcf} \
            -n 10 \
            -e {output.h5} \
            -s ref alt logit_ref logit_alt diff \
            --std_var_id
        """
"""
cd ~/.kipoi/models/DeepBind/Homo_sapiens/RBP/D00084.001_RNAcompete_A1CF
kipoi predict DeepBind/Homo_sapiens/RBP/D00084.001_RNAcompete_A1CF \
  --dataloader_args='{'intervals_file': 'example_files/intervals.bed', 'fasta_file': 'example_files/hg38_chr22.fa'}' \
  -o '/tmp/DeepBind|Homo_sapiens|RBP|D00084.001_RNAcompete_A1CF.example_pred.tsv'
# check the results
head '/tmp/DeepBind|Homo_sapiens|RBP|D00084.001_RNAcompete_A1CF.example_pred.tsv'
"""