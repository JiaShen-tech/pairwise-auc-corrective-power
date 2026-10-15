from pathlib import Path

import torch

from corrective_auc.diagnostics import endpoint_class, hoeffding_summary
from corrective_auc.losses import binary_pairwise_loss, pairwise_loss_values, pairwise_margins, surrogate_weight
from corrective_auc.train import load_experiment, multiclass_auc


def test_plain_hinge_is_relu_margin_loss() -> None:
    margins = torch.tensor([-1.0, 0.25, 1.0, 2.0])
    actual = pairwise_loss_values(margins, "hinge", margin=1.0)
    expected = torch.tensor([2.0, 0.75, 0.0, 0.0])
    assert torch.equal(actual, expected)
    assert torch.equal(surrogate_weight(margins, "hinge", margin=1.0), torch.tensor([1.0, 1.0, 0.0, 0.0]))


def test_binary_loss_uses_all_positive_negative_pairs() -> None:
    logits = torch.tensor([2.0, 0.0, -1.0, 1.0], requires_grad=True)
    labels = torch.tensor([1, 1, 0, 0])
    scores = torch.sigmoid(logits.detach())
    margins = pairwise_margins(scores, labels)
    expected = torch.relu(0.5 - margins).mean()
    actual = binary_pairwise_loss(logits, labels, "hinge", margin=0.5)
    assert torch.allclose(actual.detach(), expected)
    actual.backward()
    assert logits.grad is not None


def test_missing_class_returns_differentiable_zero() -> None:
    logits = torch.tensor([1.0, 2.0], requires_grad=True)
    loss = binary_pairwise_loss(logits, torch.ones(2), "squared_hinge")
    loss.backward()
    assert loss.item() == 0.0
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_hoeffding_decomposition_summary() -> None:
    summary = hoeffding_summary(torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64))
    assert summary["projected_drift"] == 2.5
    assert summary["v1"] > 0.0
    assert summary["v2"] == 0.0


def test_intervention_endpoint_rules() -> None:
    assert endpoint_class(0.5, -0.5) == "rescue"
    assert endpoint_class(-0.5, 0.5) == "harm"
    assert endpoint_class(0.1, -0.1) == "no_clear_effect"


def test_multiclass_metrics_use_pairwise_class_scores() -> None:
    labels = torch.tensor([0, 0, 1, 1, 2, 2]).numpy()
    scores = torch.tensor(
        [[3.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, 2.0]]
    ).numpy()
    ovo, ovr = multiclass_auc(labels, scores)
    assert ovo == 1.0
    assert ovr == 1.0


def test_experiment_config_loading() -> None:
    root = Path(__file__).resolve().parents[1]
    config, _ = load_experiment(root / "configs" / "paper.yaml", "cv_aptos_resnet50")
    assert config["model"] == "resnet50"
    assert config["task_type"] == "binary"

