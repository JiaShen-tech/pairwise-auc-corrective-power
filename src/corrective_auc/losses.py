from __future__ import annotations

import torch
import torch.nn.functional as F


PAIRWISE_LOSSES = ("square", "hinge", "exponential", "logistic", "squared_hinge")


def pairwise_margins(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return all positive-minus-negative score margins in a binary batch."""
    scores = scores.reshape(-1).float()
    labels = labels.reshape(-1).long()
    if scores.numel() != labels.numel():
        raise ValueError("scores and labels must contain the same number of elements")
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if positives.numel() == 0 or negatives.numel() == 0:
        return scores.new_empty((0, 0))
    return positives[:, None] - negatives[None, :]


def pairwise_loss_values(
    margins: torch.Tensor,
    loss: str,
    margin: float = 1.0,
) -> torch.Tensor:
    """Return a surrogate loss for each positive-negative score margin."""
    loss = loss.lower().replace("-", "_")
    if loss == "square":
        return (float(margin) - margins).square()
    if loss == "hinge":
        return F.relu(float(margin) - margins)
    if loss == "exponential":
        return torch.exp((-margins).clamp(max=20.0))
    if loss == "logistic":
        return F.softplus(-margins)
    if loss == "squared_hinge":
        return F.relu(float(margin) - margins).square()
    raise ValueError(f"unknown pairwise loss: {loss}")


def surrogate_weight(
    margins: torch.Tensor,
    loss: str,
    margin: float = 1.0,
) -> torch.Tensor:
    """Return ``-d loss(x) / dx`` for the five paper surrogates."""
    loss = loss.lower().replace("-", "_")
    if loss == "square":
        return 2.0 * (float(margin) - margins)
    if loss == "hinge":
        return (margins < float(margin)).to(margins.dtype)
    if loss == "exponential":
        return torch.exp((-margins).clamp(max=20.0))
    if loss == "logistic":
        return torch.sigmoid(-margins)
    if loss == "squared_hinge":
        return 2.0 * F.relu(float(margin) - margins)
    raise ValueError(f"unknown pairwise loss: {loss}")


def binary_pairwise_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss: str,
    margin: float = 1.0,
) -> torch.Tensor:
    """Pairwise loss on sigmoid scores, matching the experiment logging scale."""
    scores = torch.sigmoid(logits.reshape(-1).float())
    margins = pairwise_margins(scores, labels)
    if margins.numel() == 0:
        return scores.sum() * 0.0
    return pairwise_loss_values(margins, loss, margin).mean()


def macro_pairwise_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss: str,
    margin: float = 1.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Macro one-vs-rest pairwise loss for multiclass or multilabel data."""
    if logits.ndim != 2:
        raise ValueError("macro pairwise loss expects logits with shape [batch, classes]")
    if labels.ndim == 1:
        targets = F.one_hot(labels.long(), num_classes=logits.shape[1]).float()
        scores = logits.float()
        valid = torch.ones_like(targets, dtype=torch.bool)
    elif labels.shape == logits.shape:
        targets = labels.float()
        scores = torch.sigmoid(logits.float())
        valid = torch.ones_like(targets, dtype=torch.bool) if mask is None else mask.bool()
    else:
        raise ValueError("labels must be class indices or a [batch, classes] matrix")

    losses: list[torch.Tensor] = []
    for class_index in range(logits.shape[1]):
        keep = valid[:, class_index]
        class_targets = targets[keep, class_index].long()
        class_scores = scores[keep, class_index]
        margins = pairwise_margins(class_scores, class_targets)
        if margins.numel():
            losses.append(pairwise_loss_values(margins, loss, margin).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def pointwise_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    task_type: str,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if task_type == "binary":
        return F.binary_cross_entropy_with_logits(logits.reshape(-1).float(), labels.reshape(-1).float())
    if task_type == "multiclass":
        return F.cross_entropy(logits.float(), labels.long())
    if task_type == "multilabel":
        raw = F.binary_cross_entropy_with_logits(logits.float(), labels.float(), reduction="none")
        valid = torch.ones_like(raw, dtype=torch.bool) if mask is None else mask.bool()
        if not valid.any():
            return logits.sum() * 0.0
        return raw[valid].mean()
    raise ValueError(f"unknown task type: {task_type}")


def training_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    task_type: str,
    loss: str,
    margin: float = 1.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if loss == "ce":
        return pointwise_loss(logits, labels, task_type, mask)
    if task_type == "binary":
        return binary_pairwise_loss(logits, labels, loss, margin)
    return macro_pairwise_loss(logits, labels, loss, margin, mask)

