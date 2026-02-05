# ChemProFlow

## Install

```
conda create -n chemproflow natsort optuna pandas rdkit tqdm
conda activate chemproflow
pip install iterative-stratification owlready2 pybiopax rdflib scikit-multilearn tensorboard lightning torch torchmetrics
pip install --no-deps -e .
```

## Download data

Data are available at: [10.57745/QXBLVM](https://doi.org/10.57745/QXBLVM)

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
        --input-model-transport-pkl $datadir/transport_vs_unlabeled/kfold-4/base_epoch=16.ckpt \
        --input-encoder-transport-pkl $datadir/transport_vs_unlabeled/encoder.pkl \
        --input-dirichlet-calibrator-pkl $datadir/transport_vs_unlabeled/kfold-4/dirichlet_calibrator.pkl \
        # Predict transport mechanism
        --input-dataset-tcid-csv $datadir/dataset/tcid_vs_smiles.csv \
        --input-model-tcid-pkl $datadir/tcid_vs_smiles/kfold-4/base_epoch=19.ckpt \
        --input-encoder-tcid-pkl $datadir/tcid_vs_smiles/encoder.pkl \
        --input-threshold-tcid-json $datadir/tcid_vs_smiles/kfold-4/thresholds.json \
        # Retrieve microorganisms from TC-ID
        --input-catalog-micro-organisms-csv $datadir/biocyc/catalog.csv \
        --input-tcid-equivalent-json $datadir/dataset/tcid_vs_smiles.json \
        --output-resuts-csv $datadir/results.csv
```
