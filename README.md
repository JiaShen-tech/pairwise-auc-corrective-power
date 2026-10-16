# Pairwise AUC Optimization Needs Corrective Power: A Unified View

[Chinese](README_zh.md)

[Paper](https://arxiv.org/abs/2610.21743) | [PDF](https://arxiv.org/pdf/2610.21743)

Code and configurations for the experiments in the paper.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install the optional dependencies required by NLP or Deep-AUC experiments:

```bash
pip install -e ".[nlp]"
pip install -e ".[deepauc]"
```

Use a PyTorch build compatible with the CUDA version on the training machine.

## Pairwise losses

Let $m=s^+-s^-$ be the positive-minus-negative score margin and let $\gamma$ be the target margin.

| Loss | Definition |
| --- | --- |
| Square | $(\gamma-m)^2$ |
| Hinge | $[\gamma-m]_+$ |
| Exponential | $\exp(-m)$ |
| Logistic | $\log(1+\exp(-m))$ |
| Squared hinge | $[\gamma-m]_+^2$ |

Binary and multilabel tasks form margins from sigmoid scores. Multiclass tasks use class scores and a macro one-vs-rest pairwise objective.

## Data

Prepare a CSV manifest for each dataset. Paths are resolved relative to `--data-root`. Standard experiments use `train`, `val`, and `test` splits. The fixed-budget iNaturalist 2018 experiment uses `train` and `test`, where `test` is the official validation set.

Image manifest:

```csv
split,path,label
train,images/example_001.jpg,1
val,images/example_002.jpg,0
test,images/example_003.jpg,1
```

Text manifest (`text_pair` is optional):

```csv
split,text,text_pair,label
train,"premise text","hypothesis text",1
val,"premise text","hypothesis text",0
test,"premise text","hypothesis text",1
```

For multilabel tasks, use the label columns declared in `configs/paper.yaml`. An optional `mask_<label>` column marks valid targets. Under the CheXpert U-Zero setting, uncertain and missing labels are mapped to zero. Dataset sources and task definitions are also listed in `configs/paper.yaml`.

## Training

```bash
auc-train \
  --config configs/paper.yaml \
  --experiment cv_aptos_resnet50 \
  --manifest /path/to/manifests/aptos.csv \
  --data-root /path/to/datasets \
  --loss hinge \
  --margin 0.4 \
  --warmup-ratio 0.10 \
  --seed 123 \
  --output outputs/aptos/hinge_m04_w010_seed123
```

Use `--warmup-ratio 0` for direct pairwise training and `--loss ce` for the pointwise baseline. Small-scale experiments use seeds 123–127; the two large-scale experiments use the seed specified in their configuration. When a validation split is present, its AUC selects the checkpoint evaluated on the test split. The fixed-budget iNaturalist 2018 experiment evaluates the final model on the official validation set. Add `--save-checkpoint` to save model weights.

## Results

Aggregate runs with matching configurations:

```bash
auc-report outputs --output outputs/summary.csv
```

The summary reports the mean and sample standard deviation over available seeds. Hyperparameters are selected by mean validation AUC, and the associated test AUC is reported.

## Mechanism experiments

`src/corrective_auc/diagnostics.py` implements the severe-error direction, projected pair-gradient kernel, Hoeffding variance terms, batch-size scaling, parameter interventions, and endpoint classification. Their settings are under `paper.mechanism` in `configs/paper.yaml`.

Run both controlled simulations with:

```bash
auc-simulate --kind all --output outputs/simulations
```

## Tests

```bash
pip install -e ".[test]"
pytest
```

## Citation

```bibtex
@article{chen2026pairwise,
  title={Pairwise AUC Optimization Needs Corrective Power: A Unified View},
  author={Chen, Jia and Yang, Zhiyong and Bao, Shilong and Xu, Qianqian and Huang, Qingming},
  journal={arXiv preprint arXiv:2610.21743},
  year={2026}
}
```
