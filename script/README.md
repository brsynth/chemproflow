# Script

Standalone analysis scripts used to produce supplementary results and dataset
diagnostics for ChemProFlow. They are not part of the installed `chemproflow`
package and are run directly from a clone of this repository.

## Install

```bash
# Create env
conda create \
    -c conda-forge -c bioconda -c pytorch -c nvidia \
    -n chemproflow-script \
    cudatoolkit cudnn deepchem=2.8.0 dgl=2.3.0 dgllife=0.3.2 pytorch=2.3.1 pytorch_geometric=2.6.1 rdkit=2024.03.6 scikit-learn=1.6.1 tqdm
conda activate chemproflow-script
```

`select_best_kfold.py` and `select_promiscuous_tcid.py` additionally import the
`chemproflow` package (`chemproflow.utils.misc`, `chemproflow.utils.splitters`).
Make sure it is importable, e.g. by also installing it in this env:

```bash
pip install --no-deps -e ..
```

## Analyses

```bash
export datadir=<path>/chemproflow
```

### Compare transport classifiers

Run Random Forest and AttentiveFP models to predict whether molecules are
transported or not, for comparison against the D-MPNN model. The
`transport_vs_unlabeled` directory is produced by `chemproflow`'s `pu.train`
(see the [main README](../README.md)):

```bash
python ./script/compare_transport_classifiers.py \
    --input-chemproflow-str $datadir/transport_vs_unlabeled \
    --output-dir-str $datadir/comparisonmodel
```

### Select best K-Fold

Summarize per-fold performance from a `chemproflow.tcid.train` run (the
directory containing `kfold.json`) and report the best fold:

```bash
python ./script/select_best_kfold.py \
    --input-analysis-str $datadir/tcid_vs_smiles/kfold-4 \
    --output-results-csv $datadir/tcid_vs_smiles/best_kfold.csv
```

### Select promiscuous TC-IDs

From `tcid_vs_smiles.csv`, select TC-IDs suitable for a 50/50 substrate-holdout
evaluation, restricted to a GO term subtree (defaults to transmembrane
transporter activity, `GO:0042908`) and a minimum number of distinct
substrates:

```bash
python ./script/select_promiscuous_tcid.py \
    --input-dataset-csv $datadir/dataset/tcid_vs_smiles.csv \
    --input-tcdb-go-tsv tcdb/tcid_to_go.tsv \
    --input-go-obo go/go.obo \
    --output-tcid-csv $datadir/dataset/promiscuous_tcid.csv
```
