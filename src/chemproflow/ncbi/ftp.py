import argparse
import gzip
import os
from typing import Dict

from chemproflow.taxonomy.db import TaxonomyDb
from chemproflow.utils import cmd
import pandas as pd
from tqdm import tqdm
from urllib.parse import urljoin


class Ncbi(object):
    URL = "https://ftp.ncbi.nlm.nih.gov/"
    SCORE_ASSEMBLY = {
        "Chromosome": 1,
        "Contig": 2,
        "Scaffold": 3,
        "Complete Genome": 4,
    }

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super(Ncbi, self).__init__(*args, **kwargs)

    @classmethod
    def is_file_valid(cls, path):
        if os.path.isfile(path):
            try:
                with gzip.open(path, "rb") as f:
                    for _ in iter(lambda: f.read(1024 * 1024), b""):
                        pass
                return True
            except (OSError, gzip.BadGzipFile):
                return False
        return False

    def download_summary(
            self,
            file_assembly_summary_txt: str,
            file_assembly_taxonomy_csv: str,
            file_best_representative_csv: str,
            file_reference_genome_csv: str,
            tax_db: TaxonomyDb,
        ) -> None:

        summary_url = urljoin(
            self.URL, "genomes/refseq/assembly_summary_refseq.txt"
        )
        if not os.path.isfile(file_assembly_summary_txt):
            print("Download:", file_assembly_summary_txt)
            cmd.url_download(summary_url, file_assembly_summary_txt)

        df_ncbi = pd.read_csv(
            file_assembly_summary_txt,
            sep="\t",
            skiprows=1,
            low_memory=False,
        )
        print("Build lineage")
        df_tax = tax_db.build_lineage(df=df_ncbi)
        df_tax.to_csv(file_assembly_taxonomy_csv, index=False)

        print("Select models")
        df_tax = df_tax[["#assembly_accession"] + TaxonomyDb.RANKS]

        # First
        df = pd.DataFrame()
        taxids = df_ncbi["taxid"].unique()
        parts = []
        for taxid in tqdm(taxids, total=len(taxids)):
            df_sub = Ncbi.select_row(df=df_ncbi, taxid=taxid)
            if not df_sub.empty:
                parts.append(df_sub)
        df = pd.concat(parts, ignore_index=True)
        df = df.merge(df_tax, how="left", on="#assembly_accession")
        df.to_csv(file_best_representative_csv, index=False)
        # Second
        df = df_ncbi[df_ncbi["refseq_category"] == "reference genome"]
        df = df.merge(df_tax, how="left", on="#assembly_accession")
        df.to_csv(file_reference_genome_csv, index=False)

    def download_one_faa(
            self,
            assembly_accession: str,
            file_assembly_summary_csv: str,
            file_output_faa: str,
        ) -> str:
        df = pd.read_csv(file_assembly_summary_csv, low_memory=False)
        df = df[df["#assembly_accession"] == assembly_accession]

        if df.empty:
            print(f"Assembly accesion: {assembly_accession} is not found")
            return ""

        url_dir = df.iloc[0]["ftp_path"]
        url_basename = os.path.basename(url_dir)
        filename = url_basename + "_protein.faa.gz"
        url_faa = os.path.join(url_dir, filename)

        if not os.path.isfile(file_output_faa):
            cmd.url_download(url_faa, file_output_faa)
        return url_basename
    
    def to_dict(self) -> Dict[str, str]:
        return {}

    def __repr__(self) -> str:
        return str(self.to_dict())

    @classmethod
    def select_row(cls, df: pd.DataFrame, taxid: int) -> pd.DataFrame:
        df_sub = df[df["taxid"] == int(taxid)]
        if df_sub.empty:
            return pd.DataFrame()
        elif df_sub.shape[0] == 1:
            return df_sub
        else:
            df_sub_sub = df_sub[df_sub["refseq_category"] == "reference genome"]
            if df_sub_sub.shape[0] == 1:
                return df_sub_sub
            elif df_sub.shape[0] > 0:
                df_sub = df_sub.copy()
                df_sub.loc[:, "assembly_level_score"] = df_sub["assembly_level"].map(Ncbi.SCORE_ASSEMBLY)
                df_sub.loc[:, "pubmed_id"] = df_sub["pubmed_id"].apply(
                    lambda x: 0 if x == "na" else len(x.split(";"))
                )
                df_sub.sort_values(
                    [
                        "assembly_level_score",
                        "genome_size",
                        "protein_coding_gene_count",
                        "pubmed_id",
                    ],
                    ascending=[False, False, False, False],
                    inplace=True,
                )
                return df_sub.head(1)
        return pd.DataFrame()

def summary(args):
    file_taxonomy_db = args.input_taxonomy_db
    file_assembly_summary_txt = args.output_assembly_summary_txt
    file_assembly_taxonomy_csv = args.output_assembly_taxonomy_csv
    file_best_representative_csv = args.output_best_representative_csv
    file_reference_genome_csv = args.output_reference_genome_csv

    tax_db = TaxonomyDb(path=file_taxonomy_db)
    ncbi = Ncbi()
    ncbi.download_summary(
        file_assembly_summary_txt=file_assembly_summary_txt,
        file_assembly_taxonomy_csv=file_assembly_taxonomy_csv,
        file_best_representative_csv=file_best_representative_csv,
        file_reference_genome_csv=file_reference_genome_csv,
        tax_db=tax_db,
    )

def download(args):
    assembly_accesion = args.input_assembly_accession_str
    file_assembly_summary_csv = args.input_assembly_summary_csv
    file_output_faa_gz = args.output_genome_faa_gz

    ncbi = Ncbi()
    ncbi.download_one_faa(
        assembly_accession=assembly_accesion,
        file_assembly_summary_csv=file_assembly_summary_csv,
        file_output_faa=file_output_faa_gz,
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    # Summary
    par_sum = subparsers.add_parser("summary")
    par_sum.add_argument("--input-taxonomy-db", required=True, help="Taxonomy database")
    par_sum.add_argument("--output-assembly-summary-txt", required=True, help="File of assembly summary")
    par_sum.add_argument("--output-assembly-taxonomy-csv", required=True, help="File of assembly summary taxonomy")
    par_sum.add_argument("--output-best-representative-csv", required=True, help="File of assembly summmary, keeping best representative genomes")
    par_sum.add_argument("--output-reference-genome-csv", required=True, help="File of assembly summmary, keeping best \"reference genome\"")
    par_sum.set_defaults(func=summary)

    # Download
    par_dow = subparsers.add_parser("download")
    par_dow.add_argument("--input-assembly-accession-str", required=True, help="Assembly accession")
    par_dow.add_argument("--input-assembly-summary-csv", required=True, help="File of assembly summary")
    par_dow.add_argument("--output-genome-faa-gz", required=True, help="Fasta file, gzip")
    par_dow.set_defaults(func=download)

    args = parser.parse_args()
    args.func(args)