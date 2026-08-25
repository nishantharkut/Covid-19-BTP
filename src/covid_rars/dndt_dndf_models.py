from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    model_name: Literal["dndt", "dndf"]
    num_trees: int
    depth: int
    used_features_rate: float

    def __post_init__(self) -> None:
        if self.model_name not in {"dndt", "dndf"}:
            raise ValueError("model_name must be either 'dndt' or 'dndf'")
        _require_positive_integer("num_trees", self.num_trees)
        _require_positive_integer("depth", self.depth)
        _validate_feature_rate(self.used_features_rate)
        if self.model_name == "dndt" and self.num_trees != 1:
            raise ValueError("a dndt configuration must contain exactly one tree")


def _require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_feature_rate(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("used_features_rate must be a number in (0, 1]")
    if not 0.0 < float(value) <= 1.0:
        raise ValueError("used_features_rate must be in (0, 1]")


def _selected_feature_count(num_features: int, used_features_rate: float) -> int:
    _require_positive_integer("num_features", num_features)
    _validate_feature_rate(used_features_rate)
    count = int(num_features * float(used_features_rate))
    if count == 0:
        raise ValueError(
            "used_features_rate selects zero features; increase the rate or "
            "feature count"
        )
    return count


def _validate_num_classes(num_classes: int) -> None:
    if (
        isinstance(num_classes, bool)
        or not isinstance(num_classes, int)
        or num_classes < 2
    ):
        raise ValueError("num_classes must be an integer of at least 2")


def _validate_features(features: torch.Tensor, expected_features: int) -> None:
    if features.ndim != 2:
        raise ValueError(
            "features must be a two-dimensional tensor shaped [batch, features]"
        )
    if features.shape[1] != expected_features:
        raise ValueError(
            f"expected {expected_features} features, received {features.shape[1]}"
        )


def _make_feature_indices(
    *,
    num_features: int,
    count: int,
    seed: int,
    feature_indices: torch.Tensor | None,
) -> torch.Tensor:
    if feature_indices is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        return torch.randperm(num_features, generator=generator)[:count]

    if not isinstance(feature_indices, torch.Tensor):
        raise ValueError("feature_indices must be a torch.Tensor")
    if feature_indices.ndim != 1:
        raise ValueError("feature_indices must be one-dimensional")
    if feature_indices.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise ValueError("feature_indices must use an integer dtype")
    if feature_indices.numel() != count:
        raise ValueError(f"feature_indices must contain exactly {count} entries")

    indices = feature_indices.detach().to(device="cpu", dtype=torch.long).clone()
    if torch.unique(indices).numel() != count:
        raise ValueError("feature_indices must contain unique entries")
    if bool(torch.any(indices < 0)) or bool(torch.any(indices >= num_features)):
        raise ValueError(
            f"feature_indices must be within [0, {num_features - 1}]"
        )
    return indices


class NeuralDecisionTree(nn.Module):
    def __init__(
        self,
        num_features: int,
        depth: int,
        used_features_rate: float,
        num_classes: int = 2,
        seed: int = 42,
        feature_indices: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        _require_positive_integer("depth", depth)
        _validate_num_classes(num_classes)
        selected_count = _selected_feature_count(num_features, used_features_rate)

        self.num_features = num_features
        self.depth = depth
        self.num_leaves = 2**depth
        self.num_classes = num_classes
        indices = _make_feature_indices(
            num_features=num_features,
            count=selected_count,
            seed=seed,
            feature_indices=feature_indices,
        )
        self.register_buffer("feature_indices", indices)

        self.decision = nn.Linear(selected_count, self.num_leaves)
        self.leaf_logits = nn.Parameter(torch.empty(self.num_leaves, num_classes))
        nn.init.xavier_uniform_(self.decision.weight)
        nn.init.zeros_(self.decision.bias)
        nn.init.normal_(self.leaf_logits, mean=0.0, std=0.05)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _validate_features(features, self.num_features)
        selected = features.index_select(1, self.feature_indices)
        decision_probability = torch.sigmoid(self.decision(selected))
        decisions = torch.stack(
            (decision_probability, 1.0 - decision_probability), dim=2
        )

        batch_size = features.shape[0]
        path_probability = features.new_ones((batch_size, 1, 1))
        begin = 1
        end = 2
        for level in range(self.depth):
            path_probability = path_probability.reshape(batch_size, -1, 1).expand(
                -1, -1, 2
            )
            path_probability = path_probability * decisions[:, begin:end, :]
            begin = end
            end = begin + 2 ** (level + 1)

        leaf_probability = torch.softmax(self.leaf_logits, dim=1)
        return path_probability.reshape(batch_size, self.num_leaves) @ leaf_probability


class NeuralDecisionForest(nn.Module):
    def __init__(
        self,
        num_trees: int,
        num_features: int,
        depth: int,
        used_features_rate: float,
        num_classes: int = 2,
        seed: int = 42,
    ) -> None:
        super().__init__()
        _require_positive_integer("num_trees", num_trees)
        self.num_features = num_features
        self.trees = nn.ModuleList(
            NeuralDecisionTree(
                num_features=num_features,
                depth=depth,
                used_features_rate=used_features_rate,
                num_classes=num_classes,
                seed=seed + tree_index,
            )
            for tree_index in range(num_trees)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _validate_features(features, self.num_features)
        return torch.stack([tree(features) for tree in self.trees], dim=0).mean(dim=0)


class NeuralDecisionClassifier(nn.Module):
    def __init__(
        self, *, num_features: int, model_config: ModelConfig, seed: int
    ) -> None:
        super().__init__()
        _require_positive_integer("num_features", num_features)
        if not isinstance(model_config, ModelConfig):
            raise ValueError("model_config must be a ModelConfig instance")
        self.num_features = num_features
        self.normalization = nn.BatchNorm1d(
            num_features, eps=0.001, momentum=0.01
        )
        if model_config.model_name == "dndt":
            self.model: nn.Module = NeuralDecisionTree(
                num_features=num_features,
                depth=model_config.depth,
                used_features_rate=model_config.used_features_rate,
                seed=seed,
            )
        else:
            self.model = NeuralDecisionForest(
                num_trees=model_config.num_trees,
                num_features=num_features,
                depth=model_config.depth,
                used_features_rate=model_config.used_features_rate,
                seed=seed,
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _validate_features(features, self.num_features)
        return self.model(self.normalization(features))


def estimate_parameter_count(
    *,
    num_features: int,
    model_config: ModelConfig,
    num_classes: int = 2,
    include_batch_norm: bool = True,
) -> int:
    selected_count = _selected_feature_count(
        num_features, model_config.used_features_rate
    )
    _validate_num_classes(num_classes)
    leaves = 2**model_config.depth
    per_tree = leaves * (selected_count + 1 + num_classes)
    tree_count = 1 if model_config.model_name == "dndt" else model_config.num_trees
    batch_norm = 2 * num_features if include_batch_norm else 0
    return batch_norm + tree_count * per_tree


__all__ = [
    "ModelConfig",
    "NeuralDecisionClassifier",
    "NeuralDecisionForest",
    "NeuralDecisionTree",
    "estimate_parameter_count",
]
