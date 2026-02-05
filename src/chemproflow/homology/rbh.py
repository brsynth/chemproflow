import os
import subprocess
import tempfile
from typing import Dict, List

import pandas as pd


def clean_up(file_org_db):
    # clean up 
    for ext in ["pdb", "phr", "pin", "pjs", "pot", "psq", "ptf", "pto"]:
        if os.path.isfile(file_org_db + f".{ext}"):
            os.remove(file_org_db + f".{ext}")

def filter_top_hits(df, min_length=50, col_top="bitscore"):
    filtered = df[
        (df["length"] >= min_length)
    ]
    col_sort = False
    if col_top == "bitscore":
        col_sort = False
    elif col_top == "evalue":
        col_sort = True
    else:
        raise ValueError("Column to filter is unknown")
    
    top_hits = (
        filtered.sort_values(["qseqid", col_top], ascending=[True, col_sort])
        .drop_duplicates(subset="qseqid", keep="first")
    )
    return top_hits[["qseqid", "sseqid"]]

def extract_rbh(forward_df, reverse_df):
    """Return reciprocal best hits between two top-hit DataFrames."""
    merged = pd.merge(
        forward_df, reverse_df,
        left_on=["qseqid", "sseqid"],
        right_on=["sseqid", "qseqid"],
        how="inner"
    )
    # Rename for clarity
    merged = merged.rename(columns={"qseqid_x": "qseqid", "sseqid_x": "sseqid"})
    return merged[["qseqid", "sseqid"]]

def check_blast_index(path_fa) -> bool:
    basename = os.path.basename(path_fa)
    shortname = ".".join(basename.split(".")[:-1])
    if os.path.isfile(os.path.join(os.path.dirname(path_fa), shortname + ".pto")):
        return True
    return False

def makedb(path_fa, path_db):
    basename = os.path.basename(path_fa)
    shortname = ".".join(basename.split(".")[:-1])
    if os.path.isfile(os.path.join(os.path.dirname(path_fa), shortname + ".pto")):
        return os.path.join(os.path.dirname(path_fa), shortname)
    cmd = ["makeblastdb", "-in", path_fa, "-dbtype", "prot", "-out", path_db]
    ret = subprocess.run(cmd, capture_output=True)
    assert ret.returncode < 1, f"Error: {ret.stdout}\n{ret.stderr}"

def calc_coverage(start, end, length):
    if length < 1:
        return 0.0
    cov = 100 * ( ( end - start + 1 ) / length)
    return round(cov, 2)

def calc_coverage_qs(row: Dict):
    qcov = calc_coverage(start=row["qstart"], end=row["qend"], length=row["qlen"])
    scov = calc_coverage(start=row["sstart"], end=row["send"], length=row["slen"])
    return pd.Series([qcov, scov])

def run_blast(cmd_blast: List, path_fa: str, path_db: str):
    fields = ["qseqid", "sseqid", "pident", "length", "mismatch", "gapopen", "qlen", "qstart", "qend", "slen", "sstart", "send", "evalue", "bitscore", "staxid", "qcovs", "stitle"]
    with tempfile.NamedTemporaryFile() as temp_file:
        cmd = cmd_blast.copy()
        cmd += ["-db", path_db]
        cmd += ["-query", path_fa]
        cmd += ["-out", temp_file.name]
        cmd += ["-outfmt", " ".join(["6"] + fields)]
        ret = subprocess.run(cmd, capture_output=True)
        assert ret.returncode < 1, f"Error: {ret.stdout}\n{ret.stderr}"

        df = pd.read_csv(temp_file.name, names=fields, sep="\t")
        df.rename(columns={"staxid": "taxid"}, inplace=True)
    df[["qcov", "scov"]] = df.apply(calc_coverage_qs, axis=1)

    clean_up(path_db)
    return df

def run_rbh(path_genome: str, path_ref: str, args_blast: List, min_length=50, col_top="bitscore", path_fwd=None, path_rev=None):
    cmd_blast = ["blastp"] + args_blast

    # Forward
    path_genome_db = tempfile.NamedTemporaryFile()
    makedb(path_fa=path_genome, path_db=path_genome_db.name)
    df_fwd = run_blast(cmd_blast=cmd_blast, path_fa=path_ref, path_db=path_genome_db.name)
    if path_fwd:
        df_fwd.to_csv(path_fwd, index=False)
    # Reverse
    path_ref_db = ""
    if check_blast_index(path_fa=path_ref):
        basename = os.path.basename(path_ref)
        shortname = ".".join(basename.split(".")[:-1])
        path_ref_db = os.path.join(os.path.dirname(path_ref), shortname)
    else:
        path_ref_db = tempfile.NamedTemporaryFile(delete=False)
        makedb(path_fa=path_ref, path_db=path_ref_db.name)
        path_ref_db = path_ref_db.name
    df_rev = run_blast(cmd_blast=cmd_blast, path_fa=path_genome, path_db=path_ref_db)
    if path_rev:
        df_rev.to_csv(path_rev, index=False)
    # Filter for best hits
    fwd_hits = filter_top_hits(df_fwd, min_length=min_length, col_top=col_top)
    rev_hits = filter_top_hits(df_rev, min_length=min_length, col_top=col_top)

    # Extract Reciprocal Best Hits
    return extract_rbh(fwd_hits, rev_hits)