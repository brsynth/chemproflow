# Script

## Install

```bash
# Create env
conda create -n chemproflow chemproflow deepchem=2.8.0
conda activate chemproflow
```

## Analyses

```bash
# Run Random Forest and AttentiveFP models to predict if molecules are transported or not for comparison against D-MPNN
# the transport_vs_unlabeled directory was produced by chemproflow

python ./script/compare_transport_classifiers.py \
    --input-chemproflow-str $datadir/chemproflow/transport_vs_unlabeled \
    --output-dir-str $datadir/chemproflow/comparisonmodel
```
