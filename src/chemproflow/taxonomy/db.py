import argparse
import os
import re
import sqlite3
import tempfile
import zipfile
from functools import cache
from urllib.request import Request, urlopen

from tqdm import tqdm
import pandas as pd


def url_download(url: str, path: str) -> None:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(req) as src, open(path, "wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
    except Exception as e:
        raise RuntimeError(f"Download failed for {url}: {e}")


class TaxonomyDb(object):
    FTP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/"
    RANKS = [
        "strain",
        "species",
        "genus",
        "family",
        "order",
        "class",
        "phylum",
        "kingdom",
        "clade",
        "domain",
        "superkingdom",
    ]


    def __init__(self, path: str):
        self.path = path
        if not os.path.isdir(os.path.dirname(self.path)):
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        # cache
        self.df_name = pd.DataFrame()
        self.df_node = pd.DataFrame()

    def init(self):
        cursor = self.conn.cursor()
        outdir = tempfile.mkdtemp()

        # Create SQL
        cursor.execute(
            """CREATE TABLE node (
                            tax_id INTEGER PRIMARY KEY,
                            parent_id INTEGER,
                            rank TEXT
                )"""
        )
        cursor.execute(
            """CREATE TABLE name (
                            id INTEGER PRIMARY KEY,
                            label TEXT,
                            uniq TEXT,
                            classification TEXT,
                            tax_id INTEGER
                )"""
        )

        # Download
        print("Download")
        nodes_path = os.path.join(outdir, "nodes.dmp")
        names_path = os.path.join(outdir, "names.dmp")
        if os.path.isfile(nodes_path) and os.path.isfile(names_path):
            return
        os.makedirs(outdir, exist_ok=True)
        url_download(
            url=self.FTP_URL + "taxdmp.zip",
            path=os.path.join(outdir, "taxdmp.zip"),
        )
        with zipfile.ZipFile(os.path.join(outdir, "taxdmp.zip"), "r") as zip_ref:
            zip_ref.extract("nodes.dmp", outdir)
            zip_ref.extract("names.dmp", outdir)
        os.remove(os.path.join(outdir, "taxdmp.zip"))

        def _format_line(array):
            for ix in range(len(array)):
                value = array[ix]
                value = value.replace("\t|", "")
                value = value.replace('"', "")
                array[ix] = value
            return array

        # Parse files
        # nodes.dmp
        """
        tax_id                  -- node id in GenBank taxonomy database
        parent tax_id               -- parent node id in GenBank taxonomy database
        rank                    -- rank of this node (superkingdom, kingdom, ...)
        embl code               -- locus-name prefix; not unique
        division id             -- see division.dmp file
        inherited div flag  (1 or 0)        -- 1 if node inherits division from parent
        genetic code id             -- see gencode.dmp file
        inherited GC  flag  (1 or 0)        -- 1 if node inherits genetic code from parent
        mitochondrial genetic code id       -- see gencode.dmp file
        inherited MGC flag  (1 or 0)        -- 1 if node inherits mitochondrial gencode from parent
        GenBank hidden flag (1 or 0)            -- 1 if name is suppressed in GenBank entry lineage
        hidden subtree root flag (1 or 0)       -- 1 if this subtree has no sequence data yet
        comments                -- free-text comments and citations
        """
        data = []
        with open(os.path.join(outdir, "nodes.dmp")) as fd:
            for line in fd.read().splitlines():
                tab = line.split("\t|\t")
                tab = _format_line(array=tab)
                data.append(
                    dict(tax_id=int(tab[0]), parent_id=int(tab[1]), rank=tab[2])
                )
        df_nodes = pd.DataFrame(data)
        # names.dmp
        """
        tax_id                  -- the id of node associated with this name
        name_txt                -- name itself
        unique name             -- the unique variant of this name if name not unique
        name class              -- (synonym, common name, ...)
        """
        data = []
        with open(os.path.join(outdir, "names.dmp")) as fd:
            for line in fd.read().splitlines():
                tab = line.split("\t|\t")
                tab = _format_line(array=tab)
                data.append(
                    dict(
                        tax_id=int(tab[0]),
                        label=tab[1],
                        uniq=tab[2],
                        classification=tab[3],
                    )
                )
        df_names = pd.DataFrame(data)
        df_names["id"] = df_names.index.tolist()

        # Insert into sql
        print("Insert into sql")
        df_nodes.to_sql("node", self.conn, if_exists="append", index=False)
        df_names.to_sql("name", self.conn, if_exists="append", index=False)

    def read_sql(self):
        if self.df_node.empty:
            self.df_node = pd.read_sql("SELECT tax_id, parent_id, rank FROM node", self.conn)
        if self.df_name.empty:
            self.df_name = pd.read_sql("SELECT id, label, uniq, classification, tax_id  FROM name", self.conn)

    def find_taxid(self, name: str):
        cursor = self.conn.cursor()
        cursor.execute(
            """SELECT label, uniq, classification, tax_id FROM name WHERE classification IN ('synonym', 'scientific name', 'in-part', 'equivalent name', 'includes', 'common name')"""
        )

        queries = cursor.fetchall()

        def _create_tokens(name):
            pattern = r"\w+|\(.*?\)"
            name = name.replace("-", " ")
            name = name.replace("_", " ")
            name = name.replace(".", "")
            # Use re.match to find the pattern in the input string
            tokens = re.findall(pattern, name)
            ntokens = []
            for ix in range(len(tokens)):
                value = tokens[ix]
                if "(" in value or ")" in value:
                    value = value.replace("(", "")
                    value = value.replace(")", "")

                value = value.lower()
                ntokens.append(value)
            return ntokens

        tax_id_strain, tax_id_specie, tax_id_genus = -1, -1, -1
        # Match strain
        name_tokens = " ".join(name.split("_"))
        name_tokens = _create_tokens(name_tokens)
        name_tokens_specie = name_tokens[:2]
        name_tokens_genus = name_tokens[0]
        for ix, query in enumerate(queries):
            query_name = query[0]
            query_tokens = _create_tokens(query_name)
            if query_tokens == name_tokens:
                tax_id_strain = query[3]
                break
            elif (
                query_tokens[:2] == name_tokens_specie and query[2] == "scientific name"
            ):
                tax_id_specie = query[3]
            elif (
                len(query_tokens) == 1
                and query_tokens[0] == name_tokens_genus
                and query[2] == "scientific name"
            ):
                tax_id_genus = query[3]

                if name.endswith("spp."):
                    return tax_id_genus

        if tax_id_strain > -1:
            return tax_id_strain
        if tax_id_specie > -1:
            return tax_id_specie
        return tax_id_genus

    @cache
    def get_label(self, taxid: int):
        query = self.df_name[(self.df_name["tax_id"] == taxid)]
        if "scientific name" in query["classification"].tolist():
            query = query[query["classification"] == "scientific name"]
            query.reset_index(inplace=True, drop=True)
            return query.loc[0, "label"]
        elif not query.empty:
            query.reset_index(inplace=True, drop=True)
            return query.loc[0, "label"]
        return ""

    def find_lineage(self, taxid: int):
        # init
        self.read_sql()
        lineages = []

        current_taxid = taxid
        while True:
            query = self.df_node[self.df_node["tax_id"] == current_taxid]
            if query.empty:
                return lineages
            query.reset_index(inplace=True, drop=True)
            parent_id, rank = query.loc[0, "parent_id"], query.loc[0, "rank"]
            lineages.append(dict(taxid=int(current_taxid), rank=rank))
            #print(lineages, parent_id)
            current_taxid = parent_id
            if current_taxid in [1, 131567]:
                break

        # Get label for each entry
        for ix in range(len(lineages)):
            lineages[ix]["label"] = self.get_label(taxid=lineages[ix]["taxid"])

        return lineages

    def _build_lineage(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Build lineage")
        for ix, row in tqdm(df.iterrows(), total=df.shape[0]):
            lineages = self.find_lineage(taxid=row["taxid"])
            if len(lineages) < 1:
                continue
            #lineages = [lineage for lineage in lineages if lineage["rank"] in TaxonomyDb.RANKS]
            # lineages: [{'taxid': 7227, 'rank': 'species', 'label': 'Drosophila melanogaster'}, ...
            for lineage in lineages:
                if lineage["rank"] in TaxonomyDb.RANKS:
                    df.at[ix, lineage["rank"]] = int(lineage["taxid"])
        return df

    def build_lineage(self, df: pd.DataFrame) -> pd.DataFrame:
        # Expected column "taxid"
        df_taxid = pd.DataFrame(columns=["taxid"] + TaxonomyDb.RANKS)
        df_taxid["taxid"] = df["taxid"].unique().tolist()
        df_taxid = self._build_lineage(df=df_taxid)

        df = df.merge(df_taxid, on="taxid", how="left")
        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-taxonomy-db", required=True, help="Output taxonomy database"
    )
    parser.add_argument(
        "--parameter-test-bool", action="store_true", help="Wether to test database after its creation"
    )
    args = parser.parse_args()

    file_taxonomy_db = args.output_taxonomy_db
    to_test = args.parameter_test_bool

    if not os.path.isfile(file_taxonomy_db):
        tax_db = TaxonomyDb(path=file_taxonomy_db)
        tax_db.init()

    tax_db = TaxonomyDb(path=file_taxonomy_db)
    if to_test:
        #tax_db.find_taxid(name="Yersinia_enterocolitica_subsp_palearctica_PhRBD_Ye1")
        lineages = tax_db.find_lineage(taxid=1235835)
        print(lineages)
        assert len(lineages) == 9
        assert lineages == [{'taxid': 1235835, 'rank': 'species', 'label': 'Anaerotruncus sp. G3(2012)'}, {'taxid': 2641626, 'rank': 'no rank', 'label': 'unclassified Anaerotruncus'}, {'taxid': 244127, 'rank': 'genus', 'label': 'Anaerotruncus'}, {'taxid': 216572, 'rank': 'family', 'label': 'Oscillospiraceae'}, {'taxid': 186802, 'rank': 'order', 'label': 'Eubacteriales'}, {'taxid': 186801, 'rank': 'class', 'label': 'Clostridia'}, {'taxid': 1239, 'rank': 'phylum', 'label': 'Bacillota'}, {'taxid': 1783272, 'rank': 'kingdom', 'label': 'Bacillati'}, {'taxid': 2, 'rank': 'domain', 'label': 'Bacteria'}]

        print("Escherichia coli")
        lineages = tax_db.find_lineage(taxid=511145)
        print(lineages)
        df = pd.DataFrame(lineages)
        print(df)