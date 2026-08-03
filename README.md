# ChemProFlow

End-to-end mapping of membrane transport from chemical structure to microorganisms.

Given one or more SMILES, ChemProFlow predicts whether a compound is a transport
candidate, assigns a transport mechanism according to the Transporter
Classification Database (TCDB), and retrieves the microorganisms encoding the
matching transport systems.

## Install

```bash
conda create -c conda-forge -c bioconda -n chemproflow chemproflow
conda activate chemproflow
```

## Get data

Pre-built datasets and trained models are available at:
[10.57745/QXBLVM](https://doi.org/10.57745/QXBLVM).

Alternatively, the datasets can be rebuilt from source. Building the code below
requires a clone of this repository (the dataset/training scripts are not part
of the installed `chemproflow` package):

```bash
export datadir=<path>/chemproflow

# Select Rhea transporters, 32G memory
python ./src/chemproflow/dataset/rhea.py \
    --input-chebi-owl chebi/chebi.owl \
    --input-rhea-biopax-owl rhea/release_139/rhea-biopax.owl \
    --input-rhea-sprot-tsv rhea/release_139/rhea2uniprot_sprot.tsv \
    --input-rhea-trembl-tsv rhea/release_139/rhea2uniprot_trembl.tsv.gz \
    --input-tcdb-uniprot-tsv tcdb/tcid_to_uniprot.tsv \
    --output-dataset-tsv $datadir/dataset/get_substrates.rhea.tsv

# Merge TCDB and Rhea transporters, 48G memory
python ./src/chemproflow/dataset/expansion.py \
    --input-chebi-owl chebi/chebi.owl \
    --input-chemproflow-rhea-tcdb-tsv $datadir/dataset/get_substrates.rhea.tsv \
    --input-tcdb-substrates-tsv tcdb/get_substrates.tsv \
    --input-biorgroup-csv biorgroup/chebis.csv.gz \
    --output-substrates-csv $datadir/dataset/get_substrates.expansion.csv.gz

# Build tcid_vs_smiles.csv and transport_vs_unlabeled.csv, 96G memory
python ./src/chemproflow/dataset/build.py \
    --input-substrates-csv $datadir/dataset/get_substrates.expansion.csv.gz \
    --input-pubchem-sql pubchem/pubchem.sql \
    --output-tcid-csv $datadir/dataset/tcid_vs_smiles.csv \
    --output-tcid-json $datadir/dataset/tcid_vs_smiles.json \
    --output-pu-csv $datadir/dataset/transport_vs_unlabeled.csv \
    --output-expand-csv $datadir/dataset/expand.csv.gz
```

Train models:

```bash
# Model for transporter ability prediction
python ./src/chemproflow/pu/train.py \
    --input-dataset-csv $datadir/dataset/transport_vs_unlabeled.csv \
    --output-dir-str $datadir/transport_vs_unlabeled

# Model for transporter mechanisms prediction
python ./src/chemproflow/tcid/train.py \
    --input-dataset-csv $datadir/dataset/tcid_vs_smiles.csv \
    --output-dir-str $datadir/tcid_vs_smiles
```

## Run

Run pipeline. This predicts whether each compound is a transport candidate,
predicts its transport mechanism, and retrieves the microorganisms encoding
the matching TC-ID:

```bash
chemproflow pipeline \
    --input-smiles-str 'OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F' \
    --input-dataset-transport-csv $datadir/dataset/transport_vs_unlabeled.csv \
    --input-model-transport-pkl "$datadir/transport_vs_unlabeled/final_model/model_before_calibration.ckpt" \
    --input-encoder-transport-pkl $datadir/transport_vs_unlabeled/encoder.pkl \
    --input-dirichlet-calibrator-pkl $datadir/transport_vs_unlabeled/final_model/dirichlet_calibrator.pkl \
    --input-dataset-tcid-csv $datadir/dataset/tcid_vs_smiles.csv \
    --input-model-tcid-pkl "$datadir/tcid_vs_smiles/final_model/model.ckpt" \
    --input-encoder-tcid-pkl $datadir/tcid_vs_smiles/encoder.pkl \
    --input-threshold-tcid-json $datadir/tcid_vs_smiles/final_model/thresholds.json \
    --input-catalog-micro-organisms-csv $datadir/biocyc/catalog.csv \
    --input-tcid-equivalent-json $datadir/dataset/tcid_vs_smiles.json \
    --output-results-csv $datadir/results.csv
```
