from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def make_two_dimensional_data() -> dict[str, torch.Tensor]:
    seed, n = 1136848004, 12
    alpha, beta, hard_offset = 0.012, 0.64, 0.16
    nonlinear, bump_scale, rbf_scale, fourier_scale = 0.16, 0.46, 0.08, 0.15
    generator = torch.Generator(device="cpu").manual_seed(seed)
    counts = [max(2, n // 3), 2 * n, n, 4 * n]
    labels = torch.cat([torch.ones(counts[0]), -torch.ones(counts[1]), torch.ones(counts[2]), -torch.ones(counts[3])])
    groups = torch.cat([torch.full((count,), index, dtype=torch.long) for index, count in enumerate(counts)])
    hard = torch.where((groups == 0) | (groups == 1), 1.0, -1.0)

    def randn(scale: float, *shape: int) -> torch.Tensor:
        return scale * torch.randn(*shape, generator=generator)

    size = len(labels)
    offset_mu = torch.zeros(size)
    offset_mu[groups == 0], offset_mu[groups == 1] = -hard_offset, hard_offset
    b_mu = torch.zeros(size)
    b_mu[groups == 0], b_mu[groups == 1] = beta, -beta
    b_mu[groups == 2], b_mu[groups == 3] = -beta, beta
    values = {
        "labels": labels,
        "offset": offset_mu + randn(0.035, size),
        "a": 0.5 * alpha * labels + randn(max(0.004, 0.20 * alpha), size),
        "b": b_mu + randn(0.055, size),
        "c1": nonlinear * (0.55 * labels + 0.45 * hard) + randn(0.050, size),
        "c2": nonlinear * 0.90 * torch.randn(size, generator=generator),
        "c3": bump_scale * (0.50 * labels - 0.35 * hard) + randn(0.035, size),
        "p1": -math.pi + 2 * math.pi * torch.rand(size, generator=generator),
        "p2": -math.pi + 2 * math.pi * torch.rand(size, generator=generator),
        "ku": 0.48 + 1.14 * torch.rand(size, generator=generator),
        "kv": -1.05 + 2.10 * torch.rand(size, generator=generator),
        "bu": 1.70 + 0.58 * torch.randn(size, generator=generator),
        "bv": 0.02 + 0.50 * torch.randn(size, generator=generator),
        "bw_u": 0.46 + 0.78 * torch.rand(size, generator=generator),
        "bw_v": 0.34 + 0.70 * torch.rand(size, generator=generator),
    }
    n_rbf, n_fourier = 14, 8
    cu = torch.cat([torch.tensor([0.05, 0.55, 1.05, 1.65, 2.35, 3.15]), -0.15 + 4.25 * torch.rand(8, generator=generator)])
    cv = torch.cat([torch.tensor([-0.72, -0.35, 0.02, 0.30, -0.20, 0.62]), -1.00 + 2.05 * torch.rand(8, generator=generator)])
    values.update(
        {
            "rbf_cu": cu + 0.10 * torch.randn(n_rbf, generator=generator),
            "rbf_cv": cv + 0.10 * torch.randn(n_rbf, generator=generator),
            "rbf_su": 0.22 + 0.62 * torch.rand(n_rbf, generator=generator),
            "rbf_sv": 0.18 + 0.54 * torch.rand(n_rbf, generator=generator),
            "fourier_fu": 0.60 + 2.40 * torch.rand(n_fourier, generator=generator),
            "fourier_fv": -2.10 + 4.20 * torch.rand(n_fourier, generator=generator),
            "fourier_phase": -math.pi + 2 * math.pi * torch.rand(n_fourier, generator=generator),
        }
    )
    rbf_pattern = torch.stack(
        [
            labels,
            hard,
            labels * hard,
            torch.where(groups == 0, 1.0, torch.where(groups == 1, -1.0, 0.0)),
            torch.where(groups == 2, 1.0, torch.where(groups == 3, -1.0, 0.0)),
        ],
        dim=1,
    )
    coefficients = torch.randn(5, n_rbf, generator=generator)
    coefficients[0] += torch.linspace(0.9, -0.6, n_rbf)
    coefficients[2] += torch.sin(torch.linspace(0, 2.5 * math.pi, n_rbf))
    values["rbf_w"] = rbf_scale * (rbf_pattern @ coefficients / math.sqrt(5.0)) + randn(
        rbf_scale * 0.18, size, n_rbf
    )
    fourier_pattern = torch.stack([labels, hard, labels * hard, torch.randn(size, generator=generator)], dim=1)
    coefficients = torch.randn(4, n_fourier, generator=generator)
    coefficients[0] += torch.cos(torch.linspace(0, 2.0 * math.pi, n_fourier))
    coefficients[2] += torch.sin(torch.linspace(0, 3.0 * math.pi, n_fourier))
    values["fourier_w"] = fourier_scale * (fourier_pattern @ coefficients / 2.0) + randn(
        fourier_scale * 0.16, size, n_fourier
    )
    return values


def scores_and_gradients(theta: torch.Tensor, data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    u, v = theta[0], theta[1]
    z1 = data["ku"] * (u - 0.72) + data["kv"] * v + data["p1"]
    h1, dh1 = torch.tanh(z1), 1.0 - torch.tanh(z1).square()
    z2 = 1.15 * u + 0.70 * v + data["p2"]
    h2, dh2 = torch.sin(z2), torch.cos(z2)
    gu, gv = (u - data["bu"]) / data["bw_u"], (v - data["bv"]) / data["bw_v"]
    bump = torch.exp(-0.5 * (gu.square() + gv.square()))
    rbf_du, rbf_dv = u - data["rbf_cu"], v - data["rbf_cv"]
    rbf = torch.exp(-0.5 * ((rbf_du / data["rbf_su"]).square() + (rbf_dv / data["rbf_sv"]).square()))
    rbf_term = (data["rbf_w"] * rbf).sum(1)
    rbf_grad_u = (data["rbf_w"] * rbf * (-rbf_du / data["rbf_su"].square())).sum(1)
    rbf_grad_v = (data["rbf_w"] * rbf * (-rbf_dv / data["rbf_sv"].square())).sum(1)
    phase = data["fourier_fu"] * u + data["fourier_fv"] * v + data["fourier_phase"]
    fourier_term = (data["fourier_w"] * torch.sin(phase)).sum(1)
    fourier_grad_u = (data["fourier_w"] * torch.cos(phase) * data["fourier_fu"]).sum(1)
    fourier_grad_v = (data["fourier_w"] * torch.cos(phase) * data["fourier_fv"]).sum(1)
    scores = (
        data["offset"] + data["a"] * u + data["b"] * v + data["c1"] * h1 + data["c2"] * h2
        + data["c3"] * bump + rbf_term + fourier_term
    )
    grad_u = (
        data["a"] + data["c1"] * dh1 * data["ku"] + 1.15 * data["c2"] * dh2
        + data["c3"] * bump * (-(u - data["bu"]) / data["bw_u"].square()) + rbf_grad_u + fourier_grad_u
    )
    grad_v = (
        data["b"] + data["c1"] * dh1 * data["kv"] + 0.70 * data["c2"] * dh2
        + data["c3"] * bump * (-(v - data["bv"]) / data["bw_v"].square()) + rbf_grad_v + fourier_grad_v
    )
    return scores, torch.stack([grad_u, grad_v], dim=1)


def two_dimensional_field(theta: torch.Tensor, data: dict[str, torch.Tensor], method: str, margin: float = 0.8):
    scores, gradients = scores_and_gradients(theta, data)
    labels = data["labels"]
    if method == "ce":
        weights = labels / (1.0 + torch.exp(torch.clamp(labels * scores, -12.0, 12.0)))
        return (weights[:, None] * gradients).mean(0)
    positive, negative = labels > 0, labels < 0
    margins = scores[positive, None] - scores[None, negative]
    pair_gradients = gradients[positive, None, :] - gradients[None, negative, :]
    if method == "hinge":
        weights = (margins < margin).float()
    elif method == "squared_hinge":
        weights = 2.0 * torch.relu(torch.tensor(margin) - margins)
    else:
        raise ValueError(method)
    return (weights[..., None] * pair_gradients).mean((0, 1))


def empirical_auc(theta: torch.Tensor, data: dict[str, torch.Tensor]) -> float:
    scores, _ = scores_and_gradients(theta, data)
    labels = data["labels"]
    margins = scores[labels > 0, None] - scores[None, labels < 0]
    return float(((margins > 0).float() + 0.5 * (margins == 0).float()).mean())


def run_two_dimensional(output: Path) -> dict[str, float]:
    data = make_two_dimensional_data()
    starts = {"bad": (-0.450, -1.3775), "intermediate": (-0.075, -1.0450), "good": (0.750, -0.5375)}
    methods = ("ce", "hinge", "squared_hinge", "warm_hinge", "warm_squared_hinge")
    rows = []
    summary = {}
    for panel, start in starts.items():
        for method in methods:
            theta = torch.tensor(start, dtype=torch.float32)
            for step in range(361):
                rows.append({"panel": panel, "method": method, "step": step, "theta_1": float(theta[0]), "theta_2": float(theta[1]), "auc": empirical_auc(theta, data)})
                if step == 360:
                    break
                active = method
                if method.startswith("warm_"):
                    active = "ce" if step < 26 else method.removeprefix("warm_")
                eta = 0.105 if active == "ce" else 0.15
                theta = (theta + eta * two_dimensional_field(theta, data, active)).clamp(-8.0, 8.0)
            summary[f"{panel}.{method}"] = rows[-1]["auc"]
    pd.DataFrame(rows).to_csv(output / "two_dimensional_trajectories.csv", index=False)
    return summary


def raw_potential(z: np.ndarray) -> np.ndarray:
    return 0.105 * (z + 1.45) ** 2 * (z - 1.34) ** 2 + 0.18 * np.exp(-((z - 0.05) / 0.37) ** 2) - 0.075 * z + 0.018 * np.sin(2.6 * z + 0.2)


def run_noise_escape(output: Path, replicates: int) -> dict[str, float]:
    grid = np.linspace(-2.10, 2.02, 3000)
    values = raw_potential(grid)
    values = 0.04 + 0.90 * (values - values.min()) / (values.max() - values.min())
    derivative = np.gradient(values, grid)
    states_by_regime = {}
    summary = {}
    for regime, seed in (("first_order", 20260524), ("collapsed", 20260512)):
        rng = np.random.default_rng(seed)
        states = np.full(replicates, -1.42, dtype=np.float64)
        escaped = np.zeros(replicates, dtype=bool)
        recorded = [states.copy()]
        sigma = 4.5 * math.sqrt(0.025 / 8) if regime == "first_order" else 4.5 * math.sqrt(0.025) / 8
        for step in range(180):
            drift = np.interp(states, grid, derivative)
            states = np.clip(states - 0.025 * drift + sigma * rng.standard_normal(replicates), -2.05, 1.95)
            escaped |= states >= 0.5370
            if (step + 1) % 2 == 0:
                recorded.append(states.copy())
        states_by_regime[regime] = np.stack(recorded)
        summary[f"{regime}.escape_fraction"] = float(escaped.mean())
    np.savez_compressed(output / "noise_escape_states.npz", **states_by_regime, grid=grid, potential=values)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the two controlled simulations reported in the paper.")
    parser.add_argument("--kind", choices=["two_dimensional", "noise_escape", "all"], default="all")
    parser.add_argument("--output", type=Path, default=Path("outputs/simulations"))
    parser.add_argument("--replicates", type=int, default=12000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}
    if args.kind in {"two_dimensional", "all"}:
        summary.update(run_two_dimensional(args.output))
    if args.kind in {"noise_escape", "all"}:
        summary.update(run_noise_escape(args.output, args.replicates))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()

