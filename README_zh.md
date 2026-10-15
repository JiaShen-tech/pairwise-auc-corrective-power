# Pairwise AUC Optimization Needs Corrective Power: A Unified View

[English](README.md)

本文仓库提供论文实验的代码与配置。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

运行 NLP 或 Deep-AUC 实验时安装相应的可选依赖：

```bash
pip install -e ".[nlp]"
pip install -e ".[deepauc]"
```

请使用与训练机器 CUDA 版本兼容的 PyTorch。

## 成对损失

令 $m=s^+-s^-$ 表示正样本分数减去负样本分数所得的间隔，$\gamma$ 表示目标间隔。

| 损失 | 定义 |
| --- | --- |
| Square | $(\gamma-m)^2$ |
| Hinge | $[\gamma-m]_+$ |
| Exponential | $\exp(-m)$ |
| Logistic | $\log(1+\exp(-m))$ |
| Squared hinge | $[\gamma-m]_+^2$ |

二分类和多标签任务使用 sigmoid 分数构造成对间隔；多分类任务使用类别分数和宏平均 one-vs-rest 成对目标。

## 数据

为每个数据集准备 CSV 清单。清单中的路径相对于 `--data-root` 解析。常规实验使用 `train`、`val` 和 `test` 三个划分。采用固定训练预算的 iNaturalist 2018 实验使用 `train` 和 `test`，其中 `test` 表示官方验证集。

图像清单：

```csv
split,path,label
train,images/example_001.jpg,1
val,images/example_002.jpg,0
test,images/example_003.jpg,1
```

文本清单，其中 `text_pair` 为可选字段：

```csv
split,text,text_pair,label
train,"premise text","hypothesis text",1
val,"premise text","hypothesis text",0
test,"premise text","hypothesis text",1
```

多标签任务使用 `configs/paper.yaml` 中声明的标签列。可选的 `mask_<label>` 列用于标记有效目标。在 CheXpert U-Zero 设置下，不确定标签和缺失标签映射为零。数据集来源和任务定义也列于 `configs/paper.yaml`。

## 训练

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

使用 `--warmup-ratio 0` 进行直接成对训练，使用 `--loss ce` 运行逐点基线。小规模实验使用 seeds 123–127；两个大规模实验使用各自配置中指定的随机种子。存在验证集时，程序根据验证集 AUC 选择用于测试的模型。采用固定训练预算的 iNaturalist 2018 实验使用最终模型评估官方验证集。添加 `--save-checkpoint` 可保存模型权重。

## 结果

汇总相同配置的实验：

```bash
auc-report outputs --output outputs/summary.csv
```

汇总结果给出各随机种子的平均值和样本标准差。根据平均验证集 AUC 选择超参数，并报告相应的测试集 AUC。

## 机制实验

`src/corrective_auc/diagnostics.py` 实现严重错误方向、投影成对梯度核、Hoeffding 方差项、批大小缩放、参数干预和终点分类。相关设置位于 `configs/paper.yaml` 的 `paper.mechanism` 中。

运行两项受控模拟：

```bash
auc-simulate --kind all --output outputs/simulations
```

## 测试

```bash
pip install -e ".[test]"
pytest
```
