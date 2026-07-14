import argparse
import re
from collections import Counter, defaultdict, deque

import pandas as pd

from chemproflow.utils.splitters import generate_scaffold


def parse_go_obo(file_go_obo):
    """Parse go.obo into a parent->children graph and an obsolete->replacement map."""
    parents_to_children = defaultdict(list)
    obsoletes = {}
    regex = re.compile(r"(GO:\d{7})")

    current_id = None
    parents = []
    replaced_ids = []

    def flush():
        if current_id is None:
            return
        if replaced_ids:
            obsoletes[current_id] = list(replaced_ids)
        for parent in parents:
            parents_to_children[parent].append(current_id)

    with open(file_go_obo) as fd:
        for line in fd:
            line = line.strip()
            if line == "[Term]":
                flush()
                current_id = None
                parents = []
                replaced_ids = []
            elif line.startswith("id: GO:"):
                current_id = line.split("id: ")[1]
            elif line.startswith("is_a:") and current_id:
                parents.append(line.split("is_a: ")[1].split(" !")[0])
            elif line.startswith("replaced_by:"):
                match = regex.search(line)
                if match:
                    replaced_ids.append(match.group(1))
        flush()

    return parents_to_children, obsoletes


def get_all_children(go_id, parents_to_children):
    seen = set()
    to_visit = deque([go_id])
    while to_visit:
        current = to_visit.popleft()
        for child in parents_to_children.get(current, []):
            if child not in seen:
                seen.add(child)
                to_visit.append(child)
    return seen


def resolve_obsolete_go_ids(go_ids, obsoletes):
    """Add obsolete GO ids whose replacement (possibly a chain of replacements) is in go_ids."""
    resolved = set(go_ids)
    for obsolete_id, replaced_by in obsoletes.items():
        seen = set()
        queue = deque(replaced_by)
        while queue:
            candidate = queue.popleft()
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate in go_ids:
                resolved.add(obsolete_id)
                break
            queue.extend(obsoletes.get(candidate, []))
    return resolved


def profile_tcids(df_dataset, tcids):
    """Compute substrate count and scaffold diversity for each TC-ID."""
    rows = []
    count_no_support = 0
    for tcid in tcids:
        smiles = df_dataset.loc[df_dataset["tcid"] == tcid, "smiles"].unique().tolist()
        support = len(smiles)
        if support == 0:
            count_no_support += 1
            continue
        scaffolds = [generate_scaffold(s, include_chirality=True) for s in smiles]
        scaffold_counts = Counter(scaffolds)
        rows.append(
            {
                "tcid": tcid,
                "support": support,
                "count_scaffold": len(scaffold_counts),
                "max_scaffold_share": max(scaffold_counts.values()) / support,
            }
        )
    print("TC-IDs with no support:", count_no_support)
    return pd.DataFrame(rows)


def select_tcids(df_count, min_support):
    """Rank TC-IDs by chemical breadth, keeping only those where a 50/50 chemically
    stratified split is meaningful (enough substrates)."""
    feasible = df_count[df_count["support"] >= min_support].copy()
    feasible = feasible.sort_values(
        ["count_scaffold", "support"], ascending=False
    ).reset_index(drop=True)

    selected_tcids = []
    for tcid in feasible["tcid"]:
        if tcid not in selected_tcids:
            selected_tcids.append(tcid)

    df_selected = df_count[df_count["tcid"].isin(selected_tcids)].copy()
    df_selected["tcid"] = pd.Categorical(
        df_selected["tcid"], categories=selected_tcids, ordered=True
    )
    return df_selected.sort_values("tcid").reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dataset-csv", required=True, help="tcid_vs_smiles.csv file")
    parser.add_argument("--input-tcdb-go-tsv", required=True, help="tcid_to_go.tsv file")
    parser.add_argument("--input-go-obo", required=True, help="go.obo file")
    parser.add_argument("--parameter-go-root-str", default="GO:0042908", help="GO id whose descendants define the substrate/transport class of interest")
    parser.add_argument("--parameter-min-support-int", default=25, type=int, help="Minimum distinct substrates required for a meaningful 50/50 holdout")
    parser.add_argument("--output-tcid-csv", required=True, help="Output CSV of selected TC-IDs")
    args = parser.parse_args()

    print("Read dataset")
    file_dataset_csv = args.input_dataset_csv
    df_dataset = pd.read_csv(file_dataset_csv)

    print("Read TCDB")
    file_tcdb_go_tsv = args.input_tcdb_go_tsv
    df_tcdb_go = pd.read_csv(
        file_tcdb_go_tsv, sep="\t", names=["go_id", "tcid", "tcid_description"]
    )
    df_tcdb_go["tcid_family"] = df_tcdb_go["tcid"].apply(lambda x: ".".join(x.split(".")[:3]))
    df_tcdb_go["tcid_subfamily"] = df_tcdb_go["tcid"].apply(lambda x: ".".join(x.split(".")[:4]))

    print("Parse GO ontology")
    file_go_obo = args.input_go_obo
    parents_to_children, obsoletes = parse_go_obo(file_go_obo)

    go_ids = get_all_children(args.parameter_go_root_str, parents_to_children)
    go_ids.add(args.parameter_go_root_str)
    go_ids = resolve_obsolete_go_ids(go_ids, obsoletes)
    print(f"GO ids selected: {len(go_ids)}")

    df_tcdb_sub = df_tcdb_go[df_tcdb_go["go_id"].isin(go_ids)].drop_duplicates("tcid").reset_index(drop=True)
    print(f"TC-IDs annotated under {args.parameter_go_root_str}: {len(df_tcdb_sub)}")

    print("Profile candidate TC-IDs (substrate count, scaffold diversity)")
    df_count = profile_tcids(df_dataset, df_tcdb_sub["tcid"])

    print("Select TC-IDs")
    df_selected = select_tcids(df_count=df_count, min_support=args.parameter_min_support_int)
    df_selected = df_selected.merge(
        df_tcdb_sub[["tcid", "tcid_description", "tcid_family", "tcid_subfamily"]],
        on="tcid",
        how="left",
    )
    df_tcdb_go = df_tcdb_go.groupby("tcid", as_index=False).agg({"go_id": set})
    df_tcdb_go["go_ids"] = df_tcdb_go["go_id"].apply(list)
    df_selected = df_selected.merge(
        df_tcdb_go[["tcid", "go_ids"]],
        on="tcid",
        how="left",
    )

    print(df_selected)
    print("Family distribution:")
    print(df_selected["tcid_family"].value_counts())

    print(f"{len(df_selected)} TC-IDs met the feasibility thresholds")

    print("Save")
    df_selected.to_csv(args.output_tcid_csv, index=False)
    