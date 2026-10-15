from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .losses import pairwise_loss_values, pairwise_margins, surrogate_weight
from .models import forward_logits


def trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def flatten_tensors(values: list[torch.Tensor | None], parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    pieces = [torch.zeros_like(parameter).reshape(-1) if value is None else value.reshape(-1) for value, parameter in zip(values, parameters)]
    return torch.cat(pieces)


def binary_scores(model: torch.nn.Module, batch: dict[str, torch.Tensor], modality: str) -> torch.Tensor:
    return torch.sigmoid(forward_logits(model, batch, modality).reshape(-1).float())


def severe_direction(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    modality: str,
    loss: str,
    margin: float,
    threshold: float = 0.0,
    eps: float = 1e-12,
) -> torch.Tensor | None:
    """Eq. (empirical severe direction) on one fixed selection batch."""
    parameters = trainable_parameters(model)
    scores = binary_scores(model, batch, modality)
    margins = pairwise_margins(scores, batch["labels"])
    if not margins.numel():
        return None
    keep = margins <= -float(threshold)
    if not keep.any():
        return None
    objective = pairwise_loss_values(margins, loss, margin)[keep].mean()
    gradients = torch.autograd.grad(objective, parameters, allow_unused=True)
    normal = flatten_tensors(list(gradients), parameters)
    norm = torch.linalg.vector_norm(normal)
    if not torch.isfinite(norm) or norm <= eps:
        return None
    return (-normal / norm).detach()


def _dot_with_direction(
    gradients: tuple[torch.Tensor | None, ...],
    parameters: list[torch.nn.Parameter],
    direction: torch.Tensor,
) -> torch.Tensor:
    cursor = 0
    total = direction.new_zeros(())
    for gradient, parameter in zip(gradients, parameters):
        count = parameter.numel()
        if gradient is not None:
            total = total + (gradient.reshape(-1) * direction[cursor : cursor + count]).sum()
        cursor += count
    return total


def projected_kernel(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    modality: str,
    direction: torch.Tensor,
    loss: str,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Projected pair-gradient kernel on an independent logging batch."""
    parameters = trainable_parameters(model)
    scores = binary_scores(model, batch, modality)
    derivatives = []
    for index, score in enumerate(scores):
        gradients = torch.autograd.grad(score, parameters, retain_graph=index + 1 < len(scores), allow_unused=True)
        derivatives.append(_dot_with_direction(gradients, parameters, direction))
    derivatives_t = torch.stack(derivatives)
    labels = batch["labels"].reshape(-1).long()
    positive_scores, negative_scores = scores[labels == 1], scores[labels == 0]
    positive_derivatives = derivatives_t[labels == 1]
    negative_derivatives = derivatives_t[labels == 0]
    if not positive_scores.numel() or not negative_scores.numel():
        empty = scores.new_empty((0, 0))
        return empty, empty
    margins = positive_scores[:, None] - negative_scores[None, :]
    weights = surrogate_weight(margins, loss, margin)
    kernel = weights * (negative_derivatives[None, :] - positive_derivatives[:, None])
    return kernel.detach().double(), margins.detach().double()


def hoeffding_summary(kernel: torch.Tensor) -> dict[str, float]:
    if kernel.numel() == 0:
        raise ValueError("kernel must contain at least one positive-negative pair")
    mean = kernel.mean()
    row = kernel.mean(dim=1)
    column = kernel.mean(dim=0)
    residual = kernel - row[:, None] - column[None, :] + mean
    row_variance = row.var(unbiased=True) if len(row) > 1 else row.new_zeros(())
    column_variance = column.var(unbiased=True) if len(column) > 1 else column.new_zeros(())
    residual_variance = residual.reshape(-1).var(unbiased=True) if residual.numel() > 1 else mean.new_zeros(())
    return {
        "projected_drift": float(mean),
        "projected_update_size": float(mean.abs()),
        "v1": float(row_variance + column_variance),
        "v1_scaled": float(row_variance / len(row) + column_variance / len(column)),
        "v2": float(residual_variance),
        "v2_scaled": float(residual_variance / kernel.numel()),
    }


def batch_variance_scaling(
    kernel: torch.Tensor,
    batch_sizes: list[int],
    repetitions: int = 5000,
    seed: int = 0,
) -> dict[str, Any]:
    """With-replacement fixed-kernel variance scaling used by the appendix check."""
    generator = torch.Generator(device=kernel.device).manual_seed(seed)
    rows = []
    for batch_size in batch_sizes:
        estimates = []
        for _ in range(repetitions):
            positive = torch.randint(len(kernel), (batch_size,), generator=generator, device=kernel.device)
            negative = torch.randint(kernel.shape[1], (batch_size,), generator=generator, device=kernel.device)
            estimates.append(kernel[positive[:, None], negative[None, :]].mean())
        variance = torch.stack(estimates).var(unbiased=True)
        rows.append({"batch_size": int(batch_size), "variance": float(variance)})
    x = np.log([row["batch_size"] for row in rows])
    y = np.log([max(row["variance"], np.finfo(float).tiny) for row in rows])
    slope = float(np.polyfit(x, y, deg=1)[0])
    return {"slope": slope, "rows": rows}


@dataclass
class NoiseIntervention:
    mode: str
    magnitude: float
    direction: torch.Tensor | None = None

    @torch.no_grad()
    def apply(self, model: torch.nn.Module, generator: torch.Generator) -> None:
        parameters = trainable_parameters(model)
        total = sum(parameter.numel() for parameter in parameters)
        if self.mode == "directional":
            if self.direction is None or self.direction.numel() != total:
                raise ValueError("directional intervention needs one normalized trainable-parameter direction")
            device = parameters[0].device
            sign = 1.0 if torch.randint(2, (), generator=generator, device=device).item() else -1.0
            noise = self.direction.to(device) * (sign * self.magnitude)
        elif self.mode == "isotropic":
            noise = torch.randn(total, generator=generator, device=parameters[0].device)
            noise = noise / torch.linalg.vector_norm(noise).clamp_min(1e-12) * self.magnitude
        elif self.mode == "none":
            return
        else:
            raise ValueError(f"unknown intervention mode: {self.mode}")
        cursor = 0
        for parameter in parameters:
            count = parameter.numel()
            parameter.add_(noise[cursor : cursor + count].view_as(parameter))
            cursor += count


def endpoint_class(delta_auc_points: float, delta_severe_points: float) -> str:
    if delta_auc_points >= 0.5 and delta_severe_points <= -0.5:
        return "rescue"
    if delta_auc_points <= -0.5 and delta_severe_points >= 0.5:
        return "harm"
    return "no_clear_effect"

