# Multi-representation molecular activity models for PPARγ and MEK1

Code accompanying the dual-target virtual screening component of the manuscript.
Four molecular representations are each paired with seven classifiers and evaluated
under a fixed scaffold-based five-fold split; an end-to-end D-MPNN is trained
separately as an independent graph baseline.

This repository contains the model code only. It does not include datasets,
trained weights, or result tables.

## Representations and models

| Representation | Dimension | Source |
| --- | --- | --- |
| Morgan | 2048 | RDKit, radius 2, `useChirality=False`, `useFeatures=False` |
| Uni-Mol | 512 | `unimol_tools.UniMolRepr`, ETKDGv3 conformer + MMFF optimisation |
| NaFM | 256 | NaFM pretrained checkpoint, official graph featurisation |
| MolFormer | 768 | `ibm-research/MoLFormer-XL-both-10pct`, pooler output |

Classifiers: logistic regression, random forest, extra trees, RBF-SVM, XGBoost,
LightGBM, MLP. Every classifier uses one fixed configuration shared across all
four representations and both targets, so no per-target tuning is involved.
The decision threshold for each representation-classifier pair is the value that
maximises MCC on the pooled out-of-fold predictions.

The D-MPNN baseline is Chemprop 1.6 trained with five seeds per fold; the
per-compound prediction is the median over seeds.

## Data

| Target | Endpoint | Threshold | Training sources | External set |
| --- | --- | --- | --- | --- |
| PPARγ | EC50 response | ≤ 1 µM active | ChEMBL (CHEMBL235) + BindingDB | NPASS target NPT99 |
| MEK1 / MAP2K1 | IC50 | ≤ 1 µM active | ChEMBL + BindingDB | NPASS |

Records whose relation and value cannot decide the label at the threshold are
marked uncertain and excluded from binary training. For PPARγ, ChEMBL assays
described as inverse agonist or antagonist are excluded; records without a
direction field are retained as EC50-response evidence and are therefore not
claimed to be individually confirmed agonists.

Folds are assigned by `StratifiedGroupKFold(n_splits=5, shuffle=True,
random_state=23)` grouped on Bemis-Murcko scaffolds, so no scaffold spans
two folds. Exact canonical-structure overlaps with the external set are removed
from training before folds are assigned.

Source files must be downloaded by the user:

- ChEMBL activity export per target, tab separated
- BindingDB activity export per target, tab separated
- `NPASS3.0_activities.txt` and `NPASS3.0_naturalproducts_structure.txt`

## Environments

The four representations have mutually incompatible dependencies and were run in
separate conda environments.

```
# classifiers and benchmark
numpy 2.2.6  pandas 2.3.3  scikit-learn 1.7.2  lightgbm 4.7.0
xgboost 3.2.0  scipy 1.15.3  rdkit 2026.3.3

# Uni-Mol            python 3.10, torch 2.10, unimol_tools
# NaFM               python 3.10, torch 2.10, torch-geometric 2.6.1
# MolFormer          python 3.10, torch 2.4.1, transformers
# D-MPNN             python 3.10, chemprop 1.6.1
```

The NaFM checkpoint and the MolFormer weights are obtained from their own
upstream releases and are not redistributed here.

## Usage

```bash
# 1. build a labelled dataset with scaffold folds
python src/prepare_dataset.py \
    --target PPARG --endpoint EC50 --threshold-um 1.0 \
    --chembl chembl-pparg.tsv --bindingdb bindingdb-pparg.tsv \
    --npass-activities NPASS3.0_activities.txt \
    --npass-structures NPASS3.0_naturalproducts_structure.txt \
    --npass-target-id NPT99 --chembl-target-id CHEMBL235 \
    --out data/pparg_ec50_1uM

# 2. featurise, one environment per representation
python src/featurize_morgan.py    --input data/pparg_ec50_1uM/training.csv --out features/pparg/morgan
python src/featurize_unimol.py    --input data/pparg_ec50_1uM/training.csv --out features/pparg/unimol
python src/featurize_nafm.py      --input data/pparg_ec50_1uM/training.csv --out features/pparg/nafm --nafm-repo /path/to/NaFM-Official
python src/featurize_molformer.py --input data/pparg_ec50_1uM/training.csv --out features/pparg/molformer

# 3. representation x classifier benchmark under the locked folds
python src/benchmark_representations.py \
    --master data/pparg_ec50_1uM/training.csv \
    --features features/pparg --out runs/pparg_benchmark

# 4. independent D-MPNN baseline
python src/train_dmpnn.py --data data/pparg_ec50_1uM/training.csv --out runs/pparg_dmpnn

# 5. score new compounds with a frozen representation-classifier pair
python src/predict_new_compounds.py \
    --master data/pparg_ec50_1uM/training.csv \
    --features features/pparg --representation Morgan --classifier ExtraTrees \
    --query-features features/query/morgan --query query.csv \
    --out predictions.csv
```

Every script takes explicit paths; nothing is hard coded to a local directory.

## Notes on the implementation

RBF-SVM is fitted with `probability=False` to avoid a second internal Platt
calibration layer on top of the outer cross-validation. The monotonic decision
score is mapped through a logistic function only so that all models share a
`[0, 1]` output. Ranking metrics are invariant to that transform and thresholds
are derived from out-of-fold predictions regardless.

Uni-Mol requires 3D coordinates. Conformers are generated with ETKDGv3 at a
fixed per-row seed followed by MMFF optimisation where it converges. Compounds
whose conformer generation fails are recorded in a status file rather than
dropped silently.

## Licence

MIT, see `LICENSE`.
