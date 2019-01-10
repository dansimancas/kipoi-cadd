rule annotate_vcf:
    input:
        infile = "/s/project/kipoi-cadd/data/raw/v1.4/validation/clinVar-ExAC/clinvar_20180729_pathogenic_all_GRCh37.vcf",
        cadd_folder = "~/Projects/CADD_scripts/"
    output:
        outfile = "/s/project/kipoi-cadd/data/raw/v1.4/validation/clinVar-ExAC/clinvar_20180729_pathogenic_all_GRCh37.tsv"
    shell:
        "./CADD_annotate.sh -c {input.cadd_folder} -o {output.outfile} -g GRCh37 -a {input.infile}"


rule predict:
    input:
        parq = 'data/processed/kipoi/chr{chr}/variant_labels.parq'
    output:
        vcf = temp('data/processed/kipoi/chr{chr}/variants.vcf')
    run:
        from fastparquet import ParquetFile
        from m_kipoi.config import VCF_HEADER  # hg19 based
        from collections import OrderedDict
        
        pf = ParquetFile(input.parq)
        df = pf.to_pandas()
        df.sort_values(by='Pos', inplace=True)
        with open(output.vcf, "w") as f:
            f.write(VCF_HEADER)

        # Append the variants
        pd.DataFrame(OrderedDict([("#CHROM", df.Chrom.astype(str)),
                                  ("POS", df.Pos),
                                  ("ID", df.index.values),
                                  ("REF", df.Ref),
                                  ("ALT", df.Alt),
                                  ("QUAL", "."),
                                  ("FILTER", "."),
                                  ("INFO", "."),
                                  ])).to_csv(output.vcf, mode='a', header=True, index=False, sep="\t")


rule postprocess:
    """Tabix the vcf
    """
    input:
        vcf = "data/processed/kipoi/chr{chr}/variants.vcf"
    output:
        vcf_gz = "data/processed/kipoi/chr{chr}/variants.vcf.gz",
    shell:
        """
        # Sort the vcf file
        bgzip -c {input.vcf} > {output.vcf_gz}
        tabix -f -p vcf {output.vcf_gz}
        """
    
