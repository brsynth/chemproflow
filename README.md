# End-to-end mapping of membrane transport from chemical structure to microorganisms

## Install

```bash
# Create env
conda create -n chemproflow jinja2 natsort optuna pandas rdkit tqdm seaborn
conda activate chemproflow
pip install cmap iterative-stratification owlready2 pybiopax rdflib scikit-multilearn tensorboard lightning torch  torch_geometric torchmetrics

# Install ChemProFlow
git clone git@github.com:brsynth/chemproflow.git
pip install --no-deps -e .
```

## Get data

Data are available at: [10.57745/QXBLVM](https://doi.org/10.57745/QXBLVM)  
To build datasets available from the link above:

```bash
# Select Rhea transporters, 32G memory
python ./src/chemproflow/dataset/rhea.py \
    --input-chebi-owl chebi/chebi.owl \
    --input-rhea-biopax-owl rhea/release_139/rhea-biopax.owl \
    --input-rhea-sprot-tsv rhea/release_139/rhea2uniprot_sprot.tsv \
    --input-rhea-trembl-tsv rhea/release_139/rhea2uniprot_trembl.tsv.gz \
    --input-tcdb-uniprot-tsv tcdb/tcid_to_uniprot.tsv \
    --output-dataset-tsv chemproflow/dataset/get_substrates.rhea.tsv

# Merge TCDB and Rhea transporters, 48G memory
python ./src/chemproflow/dataset/expansion.py \
    --input-chebi-owl chebi/chebi.owl \
    --input-chemproflow-rhea-tcdb-tsv chemproflow/dataset/get_substrates.rhea.tsv \
    --input-tcdb-substrates-tsv tcdb/get_substrates.tsv \
    --input-biorgroup-csv biorgroup/chebis.csv.gz \
    --output-substrates-csv chemproflow/dataset/get_substrates.expansion.csv.gz

# Build tcid_vs_smiles.csv and transport_vs_unlabeled.csv, 96G memory
python ./src/chemproflow/dataset/build.py \
    --input-substrates-csv chemproflow/dataset/get_substrates.expansion.csv.gz \
    --input-pubchem-sql pubchem/pubchem.sql \
    --output-tcid-csv chemproflow/dataset/tcid_vs_smiles.csv \
    --output-tcid-json chemproflow/dataset/tcid_vs_smiles.json \
    --output-pu-csv chemproflow/dataset/transport_vs_unlabeled.csv \
    --output-expand-csv $datadir/chemproflow/dataset/expand.csv.gz
```

Train models:

```bash
# Model for transporter ability prediction
python ./src/chemproflow/pu/train.py \
    --input-dataset-csv chemproflow/dataset/transport_vs_unlabeled.csv \
    --output-dir-str chemproflow/transport_vs_unlabeled

# Model for transporter mechanisms prediction
python ./src/chemproflow/tcid/train.py \
    --input-dataset-csv chemproflow/dataset/tcid_vs_smiles.csv \
    --output-dir-str chemproflow/tcid_vs_smiles
```

## Run

Build input file

```bash
export datadir=<path>/chemproflow
echo 'name,smiles\nfluconazole,OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F' > $datadir/input_smiles.csv
```

Run pipeline

```bash
python chemproflow/src/chemproflow/pipeline/run.py \
        --input-smiles-csv $datadir/input_smiles.csv \
        # Predict if it's a candidate transport
        --input-dataset-transport-csv $datadir/dataset/transport_vs_unlabeled.csv \
        --input-model-transport-pkl "$datadir/transport_vs_unlabeled/kfold-4/base_epoch=16.ckpt" \
        --input-encoder-transport-pkl $datadir/transport_vs_unlabeled/encoder.pkl \
        --input-dirichlet-calibrator-pkl $datadir/transport_vs_unlabeled/kfold-4/dirichlet_calibrator.pkl \
        # Predict transport mechanism
        --input-dataset-tcid-csv $datadir/dataset/tcid_vs_smiles.csv \
        --input-model-tcid-pkl "$datadir/tcid_vs_smiles/kfold-4/base_epoch=19.ckpt" \
        --input-encoder-tcid-pkl $datadir/tcid_vs_smiles/encoder.pkl \
        --input-threshold-tcid-json $datadir/tcid_vs_smiles/kfold-4/thresholds.json \
        # Retrieve microorganisms from TC-ID
        --input-catalog-micro-organisms-csv $datadir/biocyc/catalog.csv \
        --input-tcid-equivalent-json $datadir/dataset/tcid_vs_smiles.json \
        --output-resuts-csv $datadir/results.csv
```
