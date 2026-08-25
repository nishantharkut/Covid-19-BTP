from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from covid_rars.dndt_dndf_models import (
    KerasBatchNorm1d,
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
    assert classifier.normalization.momentum == pytest.approx(0.99)


def _keras_batch_normalization_step(
    features: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    *,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    momentum: float = 0.99,
    eps: float = 0.001,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_mean = features.mean(dim=0)
    batch_var = ((features - batch_mean) ** 2).mean(dim=0)
    normalized = (features - batch_mean) / torch.sqrt(batch_var + eps)
    normalized = normalized * gamma + beta
    next_mean = momentum * running_mean + (1.0 - momentum) * batch_mean
    next_var = momentum * running_var + (1.0 - momentum) * batch_var
    return normalized, next_mean, next_var


def test_batch_normalization_matches_exact_keras_training_and_eval_equations() -> None:
    config = ModelConfig("dndt", 1, 2, 1.0)
    classifier = NeuralDecisionClassifier(
        num_features=2, model_config=config, seed=11
    )
    normalization = classifier.normalization
    first_batch = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
    second_batch = torch.tensor([[2.0, 8.0], [6.0, 12.0]])
    initial_mean = torch.zeros(2)
    initial_var = torch.ones(2)
    gamma = torch.tensor([1.5, 0.5])
    beta = torch.tensor([-0.25, 0.75])
    with torch.no_grad():
        normalization.gamma.copy_(gamma)
        normalization.beta.copy_(beta)

    expected_first, mean_after_first, var_after_first = (
        _keras_batch_normalization_step(
            first_batch,
            initial_mean,
            initial_var,
            gamma=gamma,
            beta=beta,
        )
    )
    normalization.train()
    actual_first = normalization(first_batch)
    torch.testing.assert_close(actual_first, expected_first, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        normalization.running_mean, mean_after_first, atol=1e-7, rtol=1e-7
    )
    torch.testing.assert_close(
        normalization.running_var, var_after_first, atol=1e-7, rtol=1e-7
    )

    expected_second, mean_after_second, var_after_second = (
        _keras_batch_normalization_step(
            second_batch,
            mean_after_first,
            var_after_first,
            gamma=gamma,
            beta=beta,
        )
    )
    actual_second = normalization(second_batch)
    torch.testing.assert_close(actual_second, expected_second, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        normalization.running_mean, mean_after_second, atol=1e-7, rtol=1e-7
    )
    torch.testing.assert_close(
        normalization.running_var, var_after_second, atol=1e-7, rtol=1e-7
    )

    restored = NeuralDecisionClassifier(
        num_features=2, model_config=config, seed=99
    )
    restored.load_state_dict(classifier.state_dict())
    restored.eval()
    evaluation_batch = torch.tensor([[4.0, 10.0], [8.0, 18.0]])
    expected_eval = (
        (evaluation_batch - mean_after_second)
        / torch.sqrt(var_after_second + 0.001)
        * gamma
        + beta
    )

    torch.testing.assert_close(
        restored.normalization(evaluation_batch),
        expected_eval,
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(restored.normalization.running_mean, mean_after_second)
    torch.testing.assert_close(restored.normalization.running_var, var_after_second)


def test_batch_normalization_has_trainable_gamma_and_beta() -> None:
    classifier = NeuralDecisionClassifier(
        num_features=3,
        model_config=ModelConfig("dndt", 1, 2, 1.0),
        seed=11,
    )

    parameters = dict(classifier.normalization.named_parameters())
    assert set(parameters) == {"gamma", "beta"}
    assert parameters["gamma"].requires_grad
    assert parameters["beta"].requires_grad


def test_batch_normalization_rejects_inappropriate_feature_dimensions() -> None:
    normalization = NeuralDecisionClassifier(
        num_features=2,
        model_config=ModelConfig("dndt", 1, 2, 1.0),
        seed=11,
    ).normalization

    with pytest.raises(ValueError, match="two-dimensional"):
        normalization(torch.randn(2))
    with pytest.raises(ValueError, match="expected 2 features"):
        normalization(torch.randn(3, 4))


def test_batch_normalization_allows_a_deterministic_single_sample_batch() -> None:
    normalization = NeuralDecisionClassifier(
        num_features=2,
        model_config=ModelConfig("dndt", 1, 2, 1.0),
        seed=11,
    ).normalization
    normalization.train()

    output = normalization(torch.tensor([[2.0, -4.0]]))

    torch.testing.assert_close(output, torch.zeros_like(output))
    torch.testing.assert_close(
        normalization.running_mean, torch.tensor([0.02, -0.04])
    )
    torch.testing.assert_close(normalization.running_var, torch.tensor([0.99, 0.99]))


def test_batch_normalization_preserves_float64_and_autograd() -> None:
    normalization = NeuralDecisionClassifier(
        num_features=2,
        model_config=ModelConfig("dndt", 1, 2, 1.0),
        seed=11,
    ).normalization.to(dtype=torch.float64)
    features = torch.tensor(
        [[1.0, 2.0], [3.0, 8.0]], dtype=torch.float64, requires_grad=True
    )

    output = normalization(features)
    output.square().sum().backward()

    assert output.dtype == torch.float64
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert normalization.gamma.grad is not None
    assert normalization.beta.grad is not None


@pytest.mark.parametrize(
    "dtype,magnitude",
    [(torch.float16, 60_000.0), (torch.bfloat16, 10_000_000_000.0)],
)
def test_low_precision_batch_normalization_uses_float32_moments_on_cpu(
    dtype: torch.dtype, magnitude: float
) -> None:
    normalization = KerasBatchNorm1d(2).to(dtype=dtype)
    features = torch.tensor(
        [[magnitude, -magnitude], [-magnitude, magnitude]], dtype=dtype
    )
    features_float = features.float()
    expected_variance = (
        (features_float - features_float.mean(dim=0)).square().mean(dim=0)
    )

    output = normalization(features)

    assert output.dtype == dtype
    assert torch.isfinite(output).all()
    assert normalization.running_mean.dtype == torch.float32
    assert normalization.running_var.dtype == torch.float32
    assert torch.isfinite(normalization.running_mean).all()
    assert torch.isfinite(normalization.running_var).all()
    torch.testing.assert_close(
        normalization.running_var,
        0.99 * torch.ones(2) + 0.01 * expected_variance,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_large_value_float16_cuda_batch_normalization_remains_finite() -> None:
    normalization = KerasBatchNorm1d(2).to(device="cuda", dtype=torch.float16)
    features = torch.tensor(
        [[60_000.0, -60_000.0], [-60_000.0, 60_000.0]],
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )
    expected_variance = features.detach().float().square().mean(dim=0)

    output = normalization(features)
    output.float().square().mean().backward()

    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()
    assert normalization.running_mean.dtype == torch.float32
    assert normalization.running_var.dtype == torch.float32
    assert torch.isfinite(normalization.running_mean).all()
    assert torch.isfinite(normalization.running_var).all()
    torch.testing.assert_close(
        normalization.running_var,
        0.99 * torch.ones(2, device="cuda") + 0.01 * expected_variance,
        rtol=1e-5,
        atol=1e-5,
    )
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert normalization.gamma.grad is not None
    assert torch.isfinite(normalization.gamma.grad).all()


def test_zero_row_batch_is_rejected_without_mutating_running_state() -> None:
    normalization = KerasBatchNorm1d(2)
    mean_before = normalization.running_mean.clone()
    variance_before = normalization.running_var.clone()

    with pytest.raises(ValueError, match="at least one row"):
        normalization(torch.empty(0, 2))

    torch.testing.assert_close(normalization.running_mean, mean_before)
    torch.testing.assert_close(normalization.running_var, variance_before)


@pytest.mark.parametrize(
    "module",
    [
        NeuralDecisionTree(2, 2, 1.0),
        NeuralDecisionForest(2, 2, 2, 1.0),
        NeuralDecisionClassifier(
            num_features=2,
            model_config=ModelConfig("dndt", 1, 2, 1.0),
            seed=7,
        ),
    ],
)
def test_all_model_entry_points_reject_zero_row_batches(
    module: torch.nn.Module,
) -> None:
    with pytest.raises(ValueError, match="at least one row"):
        module(torch.empty(0, 2))


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
