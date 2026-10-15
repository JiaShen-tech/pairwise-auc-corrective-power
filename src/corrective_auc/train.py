from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

from .data import build_loaders
from .losses import PAIRWISE_LOSSES, training_loss
from .models import build_model, forward_logits


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_experiment(path: str | Path, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    try:
        experiment = document["experiments"][name]
    except KeyError as error:
        raise KeyError(f"experiment {name!r} is not defined in {path}") from error
    profile_name = experiment["profile"]
    config = merge(document["defaults"]["common"], document["defaults"][profile_name])
    config = merge(config, {key: value for key, value in experiment.items() if key != "profile"})
    config["experiment"] = name
    return config, document


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def scores_for_metric(logits: torch.Tensor, task_type: str) -> torch.Tensor:
    if task_type == "binary":
        return torch.sigmoid(logits.reshape(-1).float())
    if task_type == "multiclass":
        return logits.float()
    return torch.sigmoid(logits.float())


def multiclass_auc(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    class_count = scores.shape[1]
    counts = np.bincount(labels, minlength=class_count)
    macro_ovo = None
    if class_count > 100 and counts.min() > 0 and np.all(counts == counts[0]):
        count = int(counts[0])
        order = np.argsort(labels, kind="stable")
        grouped = torch.as_tensor(scores[order].reshape(class_count, count, class_count))
        total_pairs = class_count * (class_count - 1) // 2
        auc_sum = 0.0
        for start in range(0, total_pairs, 2048):
            flat = np.arange(start, min(start + 2048, total_pairs), dtype=np.int64)
            width = 2 * class_count - 1
            left = np.floor((width - np.sqrt(width * width - 8.0 * flat)) / 2.0).astype(np.int64)
            row_start = left * (2 * class_count - left - 1) // 2
            left -= flat < row_start
            row_start = left * (2 * class_count - left - 1) // 2
            right = flat - row_start + left + 1
            left_t, right_t = torch.as_tensor(left), torch.as_tensor(right)
            positive = grouped[left_t, :, left_t] - grouped[left_t, :, right_t]
            negative = grouped[right_t, :, left_t] - grouped[right_t, :, right_t]
            greater = (positive[:, :, None] > negative[:, None, :]).sum(dim=(1, 2), dtype=torch.float64)
            ties = (positive[:, :, None] == negative[:, None, :]).sum(dim=(1, 2), dtype=torch.float64)
            auc_sum += float(((greater + 0.5 * ties) / (count * count)).sum())
        macro_ovo = auc_sum / total_pairs

    ovo_values = []
    ovr_values = []
    for left in range(class_count):
        binary = (labels == left).astype(np.int64)
        if np.unique(binary).size == 2:
            ovr_values.append(float(roc_auc_score(binary, scores[:, left])))
        if macro_ovo is None:
            for right in range(left + 1, class_count):
                keep = (labels == left) | (labels == right)
                binary_pair = (labels[keep] == left).astype(np.int64)
                if np.unique(binary_pair).size == 2:
                    pair_score = scores[keep, left] - scores[keep, right]
                    ovo_values.append(float(roc_auc_score(binary_pair, pair_score)))
    if macro_ovo is None:
        macro_ovo = float(np.mean(ovo_values)) if ovo_values else None
    if macro_ovo is None or not ovr_values:
        raise ValueError("multiclass evaluation requires at least two represented classes")
    return float(macro_ovo), float(np.mean(ovr_values))


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, float]:
    model.eval()
    score_rows, label_rows, mask_rows = [], [], []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        logits = forward_logits(model, batch, config["modality"])
        score_rows.append(scores_for_metric(logits, config["task_type"]).cpu())
        label_rows.append(batch["labels"].cpu())
        if "mask" in batch:
            mask_rows.append(batch["mask"].cpu())
    scores = torch.cat(score_rows).numpy()
    labels = torch.cat(label_rows).numpy()

    if config["task_type"] == "binary":
        return {"auc": float(roc_auc_score(labels.reshape(-1), scores.reshape(-1)))}
    if config["task_type"] == "multiclass":
        macro_ovo, macro_ovr = multiclass_auc(labels, scores)
        return {
            "auc": macro_ovo,
            "macro_ovo_auc": macro_ovo,
            "macro_ovr_auc": macro_ovr,
        }

    masks = torch.cat(mask_rows).numpy().astype(bool) if mask_rows else np.ones_like(labels, dtype=bool)
    per_label = []
    for index in range(scores.shape[1]):
        keep = masks[:, index]
        if keep.any() and np.unique(labels[keep, index]).size == 2:
            per_label.append(float(roc_auc_score(labels[keep, index], scores[keep, index])))
    if not per_label:
        raise ValueError("no multilabel target has both classes in this split")
    return {"auc": float(np.mean(per_label)), "macro_auc": float(np.mean(per_label))}


def standard_optimizer(model: torch.nn.Module, config: dict[str, Any]):
    name = config["optimizer"].lower()
    parameters: Any = model.parameters()
    if config.get("feature_bank"):
        base_lr = float(config["lr"])
        parameters = [
            {"params": model.backbone.parameters(), "lr": base_lr * float(config["backbone_lr_multiplier"])},
            {"params": model.head.parameters(), "lr": base_lr},
        ]
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=float(config["lr"]),
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
    if name == "adamw":
        return torch.optim.AdamW(
            parameters, lr=float(config["lr"]), weight_decay=float(config.get("weight_decay", 0.0))
        )
    raise ValueError(f"unsupported optimizer: {name}")


def cosine_scheduler(optimizer, total_updates: int, warmup_fraction: float):
    warmup_steps = round(total_updates * warmup_fraction)

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_updates - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def deep_auc_components(model: torch.nn.Module, config: dict[str, Any], device: torch.device):
    try:
        from libauc.losses import AUCMLoss
        from libauc.optimizers import PESG
    except ImportError as error:
        raise RuntimeError("install the deepauc extra: pip install -e '.[deepauc]'") from error
    criterion = AUCMLoss(margin=float(config.get("margin", 1.0)), version="v2", device=device)
    optimizer = PESG(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        loss_fn=criterion,
        lr=float(config["lr"]),
        mode=str(config.get("mode", "sgd")),
        clip_value=float(config.get("clip_value", 1.0)),
        epoch_decay=float(config.get("epoch_decay", 0.002)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
        momentum=float(config.get("momentum", 0.9)),
        verbose=False,
        device=device,
    )
    return criterion, optimizer


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


class ClassAwareFeatureBank:
    def __init__(self, num_classes: int, capacity: int, feature_dim: int, device: torch.device) -> None:
        self.capacity = int(capacity)
        self.features = torch.empty(num_classes, self.capacity, feature_dim, device=device)
        self.counts = np.zeros(num_classes, dtype=np.int64)
        self.offsets = np.zeros(num_classes, dtype=np.int64)

    @torch.no_grad()
    def add(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        for class_id in labels.unique().tolist():
            values = features[labels == class_id].detach()
            if len(values) >= self.capacity:
                self.features[class_id].copy_(values[-self.capacity :])
                self.counts[class_id], self.offsets[class_id] = self.capacity, 0
                continue
            count, offset = len(values), self.offsets[class_id]
            first = min(count, self.capacity - offset)
            self.features[class_id, offset : offset + first].copy_(values[:first])
            if first < count:
                self.features[class_id, : count - first].copy_(values[first:])
            self.offsets[class_id] = (offset + count) % self.capacity
            self.counts[class_id] = min(self.counts[class_id] + count, self.capacity)

    def available_classes(self) -> torch.Tensor:
        return torch.as_tensor(np.flatnonzero(self.counts), device=self.features.device)

    def class_features(self, class_id: int) -> torch.Tensor:
        return self.features[class_id, : self.counts[class_id]]

    def negative_features(self, class_id: int, count: int, generator: torch.Generator) -> torch.Tensor:
        classes = self.available_classes()
        classes = classes[classes != class_id]
        if not len(classes):
            return self.features.new_empty((0, self.features.shape[-1]))
        selected = classes[torch.randint(len(classes), (count,), generator=generator, device=classes.device)]
        counts = torch.as_tensor(self.counts, device=classes.device)
        slots = (torch.rand(count, generator=generator, device=classes.device) * counts[selected]).long()
        return self.features[selected, slots]


def feature_bank_pairwise_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    bank: ClassAwareFeatureBank,
    head: torch.nn.Linear,
    loss_name: str,
    margin: float,
    config: dict[str, Any],
    generator: torch.Generator,
) -> torch.Tensor:
    classes = torch.unique(torch.cat([labels, bank.available_classes()]))
    limit = int(config["pair_classes_per_step"])
    if len(classes) > limit:
        classes = classes[torch.randperm(len(classes), generator=generator, device=classes.device)[:limit]]
    terms = []
    for value in classes:
        class_id = int(value)
        positives = torch.cat([features[labels == class_id], bank.class_features(class_id)], dim=0)
        negatives = torch.cat(
            [features[labels != class_id], bank.negative_features(class_id, int(config["negatives_per_class"]), generator)],
            dim=0,
        )
        if not len(positives) or not len(negatives):
            continue
        positives = positives[torch.randperm(len(positives), generator=generator, device=features.device)[: int(config["positives_per_class"])]]
        negatives = negatives[torch.randperm(len(negatives), generator=generator, device=features.device)[: int(config["negatives_per_class"])]]
        positive_scores = positives @ head.weight[class_id] + head.bias[class_id]
        negative_scores = negatives @ head.weight[class_id] + head.bias[class_id]
        residual = torch.relu(float(margin) - positive_scores[:, None] + negative_scores[None, :])
        terms.append(residual.mean() if loss_name == "hinge" else residual.square().mean())
    if not terms:
        return features.sum() * 0.0
    return torch.stack(terms).mean()


def train(args: argparse.Namespace) -> dict[str, Any]:
    config, document = load_experiment(args.config, args.experiment)
    if args.loss is not None:
        config["loss"] = args.loss
    if args.margin is not None:
        config["margin"] = args.margin
    if args.warmup_ratio is not None:
        config["warmup_ratio"] = args.warmup_ratio
    config["seed"] = int(args.seed)
    config["manifest"] = str(Path(args.manifest).resolve())
    config["data_root"] = str(Path(args.data_root).resolve())
    if config.get("weight_path") and not Path(config["weight_path"]).is_absolute():
        config["weight_path"] = str(Path(args.data_root) / config["weight_path"])

    seed_everything(config["seed"])
    device = choose_device(args.device)
    model, tokenizer = build_model(config, device)
    train_loader, val_loader, test_loader = build_loaders(
        config, config["manifest"], config["data_root"], config["seed"], tokenizer
    )

    loss_name = str(config["loss"])
    deep_auc = loss_name == "deep_auc"
    if deep_auc and config["task_type"] != "binary":
        raise ValueError("LibAUC AUCMLoss(v2) comparison is defined only for binary experiments")
    if deep_auc:
        config.update(document["paper"]["deep_auc"])
        if config["modality"] == "text":
            config["epochs"] = 8
        elif args.experiment == "cv_imagenet_lt_dogs_resnet50":
            config["epochs"] = 20
        else:
            config["epochs"] = 16
    criterion, optimizer = deep_auc_components(model, config, device) if deep_auc else (None, standard_optimizer(model, config))
    bank = None
    pair_generator = None
    if config.get("feature_bank") and loss_name != "ce":
        if loss_name not in {"hinge", "squared_hinge"}:
            raise ValueError("feature-bank training supports hinge and squared_hinge")
        bank = ClassAwareFeatureBank(
            int(config["num_classes"]), int(config["bank_per_class"]), 2048, device
        )
        pair_generator = torch.Generator(device=device).manual_seed(config["seed"] + 1_000_003)

    epochs = int(config["epochs"])
    accumulation = int(config.get("gradient_accumulation", 1))
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_updates = updates_per_epoch * epochs
    warmup_steps = round(total_updates * float(config.get("warmup_ratio", 0.0)))
    scheduler = None if deep_auc else cosine_scheduler(
        optimizer, total_updates, float(config.get("scheduler_warmup_fraction", 0.0))
    )
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}.get(
        str(config.get("dtype", "float32")), torch.float32
    )
    autocast_enabled = device.type == "cuda" and dtype != torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=autocast_enabled and dtype == torch.float16)

    best_value: float | None = None if val_loader is None else -math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    global_update = 0
    lr_scaled = False
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(epochs):
        model.train()
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        running_loss = 0.0
        seen_batches = 0
        for batch_index, raw_batch in enumerate(train_loader):
            if args.smoke_steps and batch_index >= args.smoke_steps:
                break
            batch = move_batch(raw_batch, device)
            if warmup_steps and global_update >= warmup_steps and not lr_scaled:
                switch_scale = float(config.get("switch_lr_scale", 1.0))
                for group in optimizer.param_groups:
                    group["lr"] *= switch_scale
                if scheduler is not None:
                    scheduler.base_lrs = [value * switch_scale for value in scheduler.base_lrs]
                lr_scaled = True
            current_loss = "ce" if global_update < warmup_steps else loss_name
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                features = None
                if bank is not None:
                    features, logits = model(batch["inputs"], return_features=True)
                else:
                    logits = forward_logits(model, batch, config["modality"])
                if deep_auc:
                    loss = criterion(
                        torch.sigmoid(logits.reshape(-1, 1).float()), batch["labels"].reshape(-1, 1).float()
                    )
                elif bank is not None and current_loss != "ce":
                    loss = feature_bank_pairwise_loss(
                        features,
                        batch["labels"],
                        bank,
                        model.head,
                        current_loss,
                        float(config.get("margin", 1.0)),
                        config,
                        pair_generator,
                    )
                else:
                    loss = training_loss(
                        logits,
                        batch["labels"],
                        config["task_type"],
                        current_loss,
                        float(config.get("margin", 1.0)),
                        batch.get("mask"),
                    )
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            running_loss += float(loss.detach())
            seen_batches += 1
            last_microbatch = batch_index + 1 == len(train_loader) or (
                args.smoke_steps and batch_index + 1 >= args.smoke_steps
            )
            if (batch_index + 1) % accumulation == 0 or last_microbatch:
                scaler.unscale_(optimizer)
                max_norm = float(config.get("max_grad_norm", 0.0))
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                if bank is not None and features is not None:
                    bank.add(features, batch["labels"])
                global_update += 1

        validation = None if val_loader is None else evaluate(model, val_loader, device, config)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": running_loss / max(1, seen_batches),
                "validation": validation,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if validation is None or validation["auc"] > best_value:
            best_value = None if validation is None else validation["auc"]
            best_epoch = epoch + 1
            best_state = cpu_state_dict(model)
        if deep_auc and (epoch + 1) in {round(epochs * 0.5), round(epochs * 0.75)}:
            optimizer.update_regularizer(decay_factor=10)

    if best_state is None:
        raise RuntimeError("training produced no validation checkpoint")
    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, device, config)
    result = {
        "experiment": args.experiment,
        "seed": config["seed"],
        "loss": loss_name,
        "warmup_ratio": float(config.get("warmup_ratio", 0.0)),
        "margin": float(config.get("margin", 1.0)),
        "best_epoch": best_epoch,
        "best_validation": best_value,
        "test": test,
        "history": history,
        "config": config,
        "paper_seed_set": config.get("reported_seeds", document["paper"]["seeds"]),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if args.save_checkpoint:
        torch.save(best_state, output / "best.pt")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one paper experiment from a user-provided data manifest.")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--loss", choices=["ce", "deep_auc", *PAIRWISE_LOSSES])
    parser.add_argument("--margin", type=float)
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--save-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()

