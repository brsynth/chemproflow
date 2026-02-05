import argparse
import gzip
import os
import shutil
import tempfile

from chemproflow.homology.rbh import run_rbh
from chemproflow.ncbi.ftp import Ncbi
from chemproflow.utils import cmd
from natsort import natsorted
import pandas as pd
from tqdm import tqdm


def catalog(df: pd.DataFrame, outdir_protein: str, file_tcdb_fasta: str, threads: int) -> pd.DataFrame:
    cmd_blast = ["-evalue", "1e-5", "-comp_based_stats", "0", "-qcov_hsp_perc", "60", "-num_threads", str(threads)]
    
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        if os.path.isfile(row["path_rbh"]):
            continue
        location_genome = row["location_genome"]
        if not os.path.isfile(location_genome):
            genome = row["genome"]
            # Download faa
            filename = genome + "_protein.faa"
            filename_gz = genome + "_protein.faa.gz"
            url_faa_gz = os.path.join(location_genome, filename_gz)
            location_genome = os.path.join(outdir_protein, filename)
            path_faa_gz = os.path.join(outdir_protein, filename_gz)
            cmd.url_download(url_faa_gz, path_faa_gz)
            # Gunzip
            with gzip.open(path_faa_gz, 'rb') as f_in:
                with open(location_genome, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        # rbh
        df_rbh = run_rbh(
            path_genome=location_genome,
            path_ref=file_tcdb_fasta,
            args_blast=cmd_blast,
            min_length=60,
            col_top="evalue",
        )
        df_rbh.to_csv(row["path_rbh"], index=False)

    datas = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        file_rbh_csv = row["path_rbh"]
        # Parse csv
        df_rbh = pd.read_csv(file_rbh_csv)
        tcids = set(df_rbh["qseqid"].apply(lambda x: x.split("|")[-1]).tolist())
        tcids = natsorted(list(tcids))
        data = dict(tcids=tcids, taxid=row["taxid"])
        if "#assembly_accession" in row:
            data.update({"accession": row["#assembly_accession"]})
        elif "genome" in row:
            data.update({"accession": row["genome"]})
        datas.append(data)
    df = pd.DataFrame(datas)
    return df


def catalog_ncbi(args):
    # Init
    file_reference_genome_csv = args.input_reference_genome_csv
    file_tcdb_fasta = args.input_tcdb_fasta
    threads = args.parameter_thread_int
    outdir_protein = args.output_protein_dir
    outdir_rbh = args.output_rbh_dir
    file_catalog_csv = args.output_catalog_csv
    
    to_delete_dir = False
    if outdir_protein:
        os.makedirs(outdir_protein, exist_ok=True)
    else:
        outdir_protein = tempfile.mkdtemp()
        to_delete_dir = True
    os.makedirs(outdir_rbh, exist_ok=True)

    df = pd.read_csv(file_reference_genome_csv)
    df.rename(columns={"ftp_path": "location_genome"}, inplace=True)

    df["genome"] = df["location_genome"].apply(lambda x: os.path.basename(x))
    df["path_rbh"] = df["genome"].apply(lambda x: os.path.join(outdir_rbh, f"{x}.rbh.csv"))
    
    df = df[(df["domain"] == 2) | (df["kingdom"] == 4751)] # Bacteria, Fungi
    df_tax = df[["taxid", "organism_name", "species", "genus", "family", "order", "class", "phylum", "kingdom", "clade", "domain", "superkingdom"]].copy()
    df_tax.drop_duplicates(inplace=True)
    df = catalog(df=df, outdir_protein=outdir_protein, file_tcdb_fasta=file_tcdb_fasta, threads=threads)
    df = df.merge(df_tax, on="taxid", how="left")
    df.to_csv(file_catalog_csv, index=False)

    # clean up
    if to_delete_dir:
        shutil.rmtree(outdir_protein)

def catalog_biocyc(args):
    # Init
    file_tcdb_fasta = args.input_tcdb_fasta
    file_biocyc_csv = args.input_biocyc_csv
    indir_biocyc = args.indir_biocyc_str
    threads = args.parameter_thread_int
    outdir_protein = args.output_protein_dir
    outdir_rbh = args.output_rbh_dir
    file_catalog_csv = args.output_catalog_csv
    
    os.makedirs(outdir_protein, exist_ok=True)
    os.makedirs(outdir_rbh, exist_ok=True)

    # Filter
    df = pd.read_csv(file_biocyc_csv)
    print("Filter")
    df = df[(df["domain"] == 2) | (df["kingdom"] == 4751)] # Bacteria, Fungi

    print("Merge Biocyc <-> Ncbi")
    for idx, row in df.iterrows():
        path_prot = os.path.join(indir_biocyc, row["biocyc"], "protseq.fasta")
        if not os.path.isfile(path_prot):
            path_prot = os.path.join(indir_biocyc, row["biocyc"], "protseq.fsa")
        path_prot_dest = ""
        if os.path.isfile(path_prot):
            path_prot_dest = os.path.join(outdir_protein, f"{row["biocyc"]}.fasta")
            shutil.copyfile(path_prot, path_prot_dest)
        df.at[idx, "location_genome"] = path_prot_dest
    df = df[df["location_genome"] != ""]
    df["genome"] = df["biocyc"].copy()
    df["path_rbh"] = df["genome"].apply(lambda x: os.path.join(outdir_rbh, f"{x}.rbh.csv"))
    df.reset_index(drop=True, inplace=True)

    df = catalog(df=df, outdir_protein=outdir_protein, file_tcdb_fasta=file_tcdb_fasta, threads=threads)
    df.to_csv(file_catalog_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    # Biocyc
    par_bio = subparsers.add_parser("biocyc")
    par_bio.add_argument("--input-tcdb-fasta", required=True, help="TCDB proteins, proteins.fasta")
    par_bio.add_argument("--input-biocyc-csv", required=True, help="Biocyc molecules, compounds.csv")
    par_bio.add_argument("--indir-biocyc-str", required=True, help="Directory biocyc, tier1-tier2")
    par_bio.add_argument("--parameter-thread-int", type=int, default=1, help="Threads to use")
    par_bio.add_argument("--output-protein-dir", required=True, help="Directory to output genome sequence")
    par_bio.add_argument("--output-rbh-dir", required=True, help="Directory to output RBH results")
    par_bio.add_argument("--output-catalog-csv", required=True, help="Output catalog")
    par_bio.set_defaults(func=catalog_biocyc)

    # Download
    par_ncb = subparsers.add_parser("ncbi")
    par_ncb.add_argument("--input-reference-genome-csv", required=True, help="File of reference genome ncbi")
    par_ncb.add_argument("--input-tcdb-fasta", required=True, help="TCDB proteins, proteins.fasta")
    par_ncb.add_argument("--parameter-thread-int", type=int, default=1, help="Threads to use")
    par_ncb.add_argument("--output-protein-dir", help="Directory to output genome sequence")
    par_ncb.add_argument("--output-rbh-dir", required=True, help="Directory to output RBH results")
    par_ncb.add_argument("--output-catalog-csv", required=True, help="Output catalog")
    par_ncb.set_defaults(func=catalog_ncbi)

    args = parser.parse_args()
    args.func(args)
