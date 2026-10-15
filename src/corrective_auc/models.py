from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class FeatureClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, num_outputs: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, num_outputs)

    def forward(self, inputs: torch.Tensor, return_features: bool = False):
        features = self.backbone(inputs)
        logits = self.head(features)
        return (features, logits) if return_features else logits


def output_count(config: dict[str, Any]) -> int:
    if config["task_type"] == "binary":
        return 1
    if config["task_type"] == "multilabel":
        return len(config["label_columns"])
    return int(config["num_classes"])


def build_image_model(config: dict[str, Any]) -> nn.Module:
    import torchvision.models as models

    name = config["model"]
    pretrained = bool(config.get("pretrained", False))
    weights: Any = "DEFAULT" if pretrained else None
    if name == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, output_count(config))
    elif name == "resnet50":
        model = models.resnet50(weights=weights)
        feature_dim = model.fc.in_features
        if config.get("feature_bank"):
            model.fc = nn.Identity()
            model = FeatureClassifier(model, feature_dim, output_count(config))
        else:
            model.fc = nn.Linear(feature_dim, output_count(config))
    elif name == "densenet121":
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, output_count(config))
    else:
        raise ValueError(f"unsupported image model: {name}")

    weight_path = config.get("weight_path")
    if weight_path:
        payload = torch.load(Path(weight_path), map_location="cpu", weights_only=True)
        state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        cleaned = {str(key).removeprefix("module."): value for key, value in state.items()}
        current = model.state_dict()
        compatible = {}
        for key, value in cleaned.items():
            candidates = (key, f"backbone.{key}")
            target = next((candidate for candidate in candidates if candidate in current and value.shape == current[candidate].shape), None)
            if target is not None:
                compatible[target] = value
        model.load_state_dict(compatible, strict=False)
    return model


def build_text_model(config: dict[str, Any], device: torch.device):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = int(config.get("max_length", 128))

    dtype_name = str(config.get("dtype", "float32"))
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    if device.type == "cpu" and dtype != torch.float32:
        dtype = torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        name,
        num_labels=output_count(config),
        torch_dtype=dtype,
        ignore_mismatched_sizes=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    targets = config.get("lora_target_modules")
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(config.get("lora_rank", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        bias="none",
        target_modules=targets,
    )
    model = get_peft_model(model, lora)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model, tokenizer


def build_model(config: dict[str, Any], device: torch.device):
    if config["modality"] == "image":
        return build_image_model(config).to(device), None
    model, tokenizer = build_text_model(config, device)
    return model.to(device), tokenizer


def forward_logits(model: nn.Module, batch: dict[str, torch.Tensor], modality: str) -> torch.Tensor:
    if modality == "image":
        return model(batch["inputs"])
    inputs = {key: value for key, value in batch.items() if key not in {"labels", "mask"}}
    return model(**inputs).logits

