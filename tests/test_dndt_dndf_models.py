from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from covid_rars.dndt_dndf_models import (
    ModelConfig,
    NeuralDecisionClassifier,
    NeuralDecisionForest,
    NeuralDecisionTree,
    estimate_parameter_count,
)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def _depth_two_numpy_probability(
    features: np.ndarray,
    *,
    feature_indices: np.ndarray,
    decision_weight: np.ndarray,
    decision_bias: np.ndarray,
    leaf_logits: np.ndarray,
) -> np.ndarray:
    selected = features[:, feature_indices]
    decisions = 1.0 / (1.0 + np.exp(-(selected @ decision_weight.T + decision_bias)))
    root = decisions[:, 1]
    left = decisions[:, 2]
    right = decisions[:, 3]
    paths = np.column_stack(
        (
            root * left,
            root * (1.0 - left),
            (1.0 - root) * right,
            (1.0 - root) * (1.0 - right),
        )
    )
    return paths @ _softmax(leaf_logits)


def _configured_depth_two_tree() -> NeuralDecisionTree:
    tree = NeuralDecisionTree(
        num_features=3,
        depth=2,
        used_features_rate=1.0,
        num_classes=2,
        seed=7,
        feature_indices=torch.tensor([0, 1, 2]),
    )
    with torch.no_grad():
        tree.decision.weight.copy_(
            torch.tensor(
                [
                    [9.0, 9.0, 9.0],
                    [0.2, -0.1, 0.4],
                    [0.3, 0.5, -0.2],
                    [-0.4, 0.1, 0.2],
                ]
            )
        )
        tree.decision.bias.copy_(torch.tensor([9.0, 0.1, -0.2, 0.3]))
        tree.leaf_logits.copy_(
            torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 0.0], [0.0, 1.0]])
        )
    return tree


def test_model_config_is_frozen_and_validates_the_model_contract() -> None:
    config = ModelConfig(
        model_name="dndf", num_trees=25, depth=11, used_features_rate=0.6
    )
    with pytest.raises(FrozenInstanceError):
        config.depth = 2  # type: ignore[misc]

    with pytest.raises(ValueError, match="model_name"):
        ModelConfig("forest", 1, 2, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="num_trees"):
        ModelConfig("dndf", 0, 2, 1.0)
    with pytest.raises(ValueError, match="exactly one tree"):
        ModelConfig("dndt", 2, 2, 1.0)
    with pytest.raises(ValueError, match="depth"):
        ModelConfig("dndt", 1, 0, 1.0)
    with pytest.raises(ValueError, match="used_features_rate"):
        ModelConfig("dndt", 1, 2, 0.0)


def test_depth_two_tree_matches_independent_numpy_oracle() -> None:
    tree = _configured_depth_two_tree()
    features = torch.tensor([[0.5, -1.0, 0.25], [-0.2, 0.7, 1.1]])

    expected = _depth_two_numpy_probability(
        features.numpy(),
        feature_indices=np.array([0, 1, 2]),
        decision_weight=tree.decision.weight.detach().numpy(),
        decision_bias=tree.decision.bias.detach().numpy(),
        leaf_logits=tree.leaf_logits.detach().numpy(),
    )

    np.testing.assert_allclose(tree(features).detach().numpy(), expected, atol=1e-6)


def test_routing_slot_zero_is_deliberately_unused() -> None:
    tree = _configured_depth_two_tree()
    features = torch.tensor([[0.5, -1.0, 0.25], [1.0, 2.0, -3.0]])
    before = tree(features).detach().clone()

    with torch.no_grad():
        tree.decision.weight[0].fill_(-1000.0)
        tree.decision.bias[0].fill_(-1000.0)

    torch.testing.assert_close(tree(features), before)


def test_tree_selects_a_deterministic_frozen_feature_subset() -> None:
    first = NeuralDecisionTree(10, 3, 0.6, seed=123)
    second = NeuralDecisionTree(10, 3, 0.6, seed=123)
    different = NeuralDecisionTree(10, 3, 0.6, seed=124)

    assert "feature_indices" in dict(first.named_buffers())
    assert first.feature_indices.requires_grad is False
    torch.testing.assert_close(first.feature_indices, second.feature_indices)
    assert not torch.equal(first.feature_indices, different.feature_indices)
    assert first.feature_indices.numel() == 6


def test_forest_checkpoint_preserves_every_selected_feature_subset() -> None:
    source = NeuralDecisionForest(3, 8, 2, 0.5, seed=31)
    target = NeuralDecisionForest(3, 8, 2, 0.5, seed=999)
    assert any(
        not torch.equal(source_tree.feature_indices, target_tree.feature_indices)
        for source_tree, target_tree in zip(source.trees, target.trees, strict=True)
    )

    target.load_state_dict(source.state_dict())

    for source_tree, target_tree in zip(source.trees, target.trees, strict=True):
        torch.testing.assert_close(
            target_tree.feature_indices, source_tree.feature_indices
        )


def test_forest_probability_is_arithmetic_mean_of_tree_probabilities() -> None:
    forest = NeuralDecisionForest(
        num_trees=2,
        num_features=4,
        depth=2,
        used_features_rate=0.75,
        seed=9,
    )
    features = torch.randn(5, 4)
    expected = torch.stack([tree(features) for tree in forest.trees]).mean(dim=0)

    torch.testing.assert_close(forest(features), expected)


def test_forest_uses_seed_plus_tree_index_for_feature_subsets() -> None:
    forest = NeuralDecisionForest(3, 12, 2, 0.5, seed=40)
    expected = [NeuralDecisionTree(12, 2, 0.5, seed=40 + index) for index in range(3)]

    for actual_tree, expected_tree in zip(forest.trees, expected, strict=True):
        torch.testing.assert_close(
            actual_tree.feature_indices, expected_tree.feature_indices
        )


@pytest.mark.parametrize("model_name,num_trees", [("dndt", 1), ("dndf", 3)])
def test_classifier_applies_batch_normalization_before_model(
    model_name: str, num_trees: int
) -> None:
    config = ModelConfig(  # type: ignore[arg-type]
        model_name=model_name,
        num_trees=num_trees,
        depth=2,
        used_features_rate=0.75,
    )
    classifier = NeuralDecisionClassifier(
        num_features=4, model_config=config, seed=5
    )
    observed: list[torch.Tensor] = []
    handle = classifier.model.register_forward_pre_hook(
        lambda _module, args: observed.append(args[0].detach().clone())
    )
    classifier.train()
    features = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 7.0, 8.0], [5.0, 8.0, 9.0, 12.0]]
    )

    classifier(features)
    handle.remove()

    torch.testing.assert_close(observed[0], classifier.normalization(features))


def test_batch_normalization_running_state_and_checkpoint_round_trip() -> None:
    config = ModelConfig("dndt", 1, 2, 1.0)
    classifier = NeuralDecisionClassifier(
        num_features=3, model_config=config, seed=11
    )
    initial_mean = classifier.normalization.running_mean.clone()
    classifier.train()
    classifier(torch.tensor([[2.0, 4.0, 6.0], [4.0, 8.0, 12.0]]))
    assert not torch.equal(classifier.normalization.running_mean, initial_mean)

    classifier.eval()
    frozen_mean = classifier.normalization.running_mean.clone()
    expected = classifier(torch.tensor([[20.0, 40.0, 60.0], [30.0, 60.0, 90.0]]))
    torch.testing.assert_close(classifier.normalization.running_mean, frozen_mean)

    restored = NeuralDecisionClassifier(num_features=3, model_config=config, seed=99)
    restored.load_state_dict(classifier.state_dict())
    restored.eval()

    torch.testing.assert_close(restored.normalization.running_mean, frozen_mean)
    torch.testing.assert_close(
        restored(torch.tensor([[20.0, 40.0, 60.0], [30.0, 60.0, 90.0]])),
        expected,
    )


def test_batch_normalization_matches_released_keras_defaults() -> None:
    classifier = NeuralDecisionClassifier(
        num_features=3,
        model_config=ModelConfig("dndt", 1, 2, 1.0),
        seed=11,
    )

    assert classifier.normalization.eps == pytest.approx(0.001)
    # Keras momentum weights the old statistic. PyTorch momentum weights the new one.
    assert classifier.normalization.momentum == pytest.approx(1.0 - 0.99)


def test_probabilities_and_nll_gradients_are_finite() -> None:
    config = ModelConfig("dndf", 3, 3, 0.75)
    classifier = NeuralDecisionClassifier(
        num_features=8, model_config=config, seed=17
    )
    classifier.train()
    features = torch.randn(6, 8)
    labels = torch.tensor([0, 1, 1, 0, 1, 0])

    probabilities = classifier(features)
    loss = F.nll_loss(torch.log(probabilities.clamp_min(1e-7)), labels)
    loss.backward()

    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(
        probabilities.sum(dim=1), torch.ones(probabilities.shape[0])
    )
    assert torch.isfinite(loss)
    gradients = [parameter.grad for parameter in classifier.parameters()]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(
        torch.isfinite(gradient).all()
        for gradient in gradients
        if gradient is not None
    )


@pytest.mark.parametrize(
    "factory,error",
    [
        (lambda: NeuralDecisionTree(0, 2, 1.0), "num_features"),
        (lambda: NeuralDecisionTree(3, 0, 1.0), "depth"),
        (lambda: NeuralDecisionTree(3, 2, 0.0), "used_features_rate"),
        (lambda: NeuralDecisionTree(3, 2, 1.1), "used_features_rate"),
        (lambda: NeuralDecisionTree(3, 2, 0.1), "selects zero"),
        (lambda: NeuralDecisionTree(3, 2, 1.0, num_classes=1), "num_classes"),
        (lambda: NeuralDecisionForest(0, 3, 2, 1.0), "num_trees"),
    ],
)
def test_invalid_model_parameters_raise_actionable_value_errors(factory, error) -> None:
    with pytest.raises(ValueError, match=error):
        factory()


@pytest.mark.parametrize(
    "indices,error",
    [
        (torch.tensor([[0, 1, 2]]), "one-dimensional"),
        (torch.tensor([0.0, 1.0, 2.0]), "integer"),
        (torch.tensor([0, 1]), "exactly 3"),
        (torch.tensor([0, 1, 1]), "unique"),
        (torch.tensor([0, 1, 3]), "within"),
    ],
)
def test_bad_feature_indices_raise_actionable_value_errors(indices, error) -> None:
    with pytest.raises(ValueError, match=error):
        NeuralDecisionTree(3, 2, 1.0, feature_indices=indices)


@pytest.mark.parametrize(
    "module",
    [
        NeuralDecisionTree(4, 2, 0.75),
        NeuralDecisionForest(2, 4, 2, 0.75),
        NeuralDecisionClassifier(
            num_features=4,
            model_config=ModelConfig("dndt", 1, 2, 0.75),
            seed=2,
        ),
    ],
)
def test_forward_rejects_bad_feature_dimensions(module: torch.nn.Module) -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        module(torch.randn(4))
    with pytest.raises(ValueError, match="expected 4 features"):
        module(torch.randn(2, 5))


def test_module_dtype_and_registered_buffers_follow_to() -> None:
    tree = NeuralDecisionTree(4, 2, 0.75, seed=12).to(dtype=torch.float64)
    features = torch.randn(3, 4, dtype=torch.float64)

    output = tree(features)

    assert output.dtype == torch.float64
    assert tree.feature_indices.device == tree.decision.weight.device
    assert tree.feature_indices.dtype == torch.long


def test_parameter_estimator_matches_small_allocated_models() -> None:
    for config in (ModelConfig("dndt", 1, 2, 0.75), ModelConfig("dndf", 3, 2, 0.75)):
        classifier = NeuralDecisionClassifier(
            num_features=4, model_config=config, seed=7
        )
        estimate = estimate_parameter_count(
            num_features=4,
            model_config=config,
            num_classes=2,
            include_batch_norm=True,
        )
        actual = sum(parameter.numel() for parameter in classifier.parameters())
        assert estimate == actual


def test_published_forest_parameter_count_is_estimated_without_allocation() -> None:
    config = ModelConfig("dndf", 25, 11, 0.6)

    estimate = estimate_parameter_count(
        num_features=800,
        model_config=config,
        num_classes=2,
        include_batch_norm=True,
    )

    per_tree = (2**11) * (int(800 * 0.6) + 1 + 2)
    assert estimate == 2 * 800 + 25 * per_tree
    assert estimate == 24_731_200
    assert estimate * 4 < 100_000_000
