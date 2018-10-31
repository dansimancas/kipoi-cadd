import pandas as pd

ALL_CHROMOSOMES = list(range(1, 23))
# ALL_CHROMOSOMES = [22]

rule all:
    input:
        #expand(('data/processed/kipoi/chr{c}' + 
        #        '/variant_labels.parq'), c=ALL_CHROMOSOMES)
        expand(('data/processed/kipoi/chr{c}' +
                '/variants.vcf.gz'), c=ALL_CHROMOSOMES)


rule write_parquet_for_chromosome:
    input:
        all_variants = ('data/processed/kipoi/all_variants'),
    output:
        parq = 'data/processed/kipoi/chr{chr}/variant_labels.parq'
    run:
        import dask.dataframe as dd
        from fastparquet import write

        parquet = dd.read_parquet(input.all_variants, index="ID")
        temp = parquet[parquet.Chrom == wildcards.chr]
        chrom_df = temp.compute()
        chrom_df.Chrom = chrom_df.Chrom.astype('int64')
        write(output.parq, chrom_df)


rule write_vcf_for_chromosome:
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


rule tabix_vcf:
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
    