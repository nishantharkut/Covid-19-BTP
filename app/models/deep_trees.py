"""
Deep Neural Decision Trees (DNDT) and Deep Neural Decision Forests (DNDF)
Implementation for tabular/acoustic representation benchmarking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepNeuralDecisionTree(nn.Module):
    """
    DNDT (Deep Neural Decision Tree - Yang et al., 2018)
    Computes soft binning per feature, constructs joint routing via tensor outer products,
    and applies a differentiable leaf layer.
    """
    def __init__(self, in_features: int, num_cutpoints: int = 1, num_classes: int = 1, temperature: float = 1.0):
        super().__init__()
        assert in_features <= 12, "DNDT input features must be <= 12 to avoid (cutpoints+1)^D memory explosion."
        self.in_features = in_features
        self.num_cutpoints = num_cutpoints
        self.temperature = temperature
        self.num_leaves = (num_cutpoints + 1) ** in_features

        init_cuts = torch.linspace(-1.0, 1.0, steps=num_cutpoints).unsqueeze(0).repeat(in_features, 1)
        self.cutpoints = nn.Parameter(init_cuts)
        self.steepness = nn.Parameter(torch.ones(in_features, num_cutpoints))
        self.leaf_weights = nn.Parameter(torch.randn(self.num_leaves, num_classes) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_in = x.unsqueeze(-1)
        cutpoints = self.cutpoints.unsqueeze(0)
        steepness = torch.abs(self.steepness.unsqueeze(0)) + 1e-4

        split_prob = torch.sigmoid(steepness * (x_in - cutpoints) / self.temperature)

        left = torch.cat([torch.ones(batch_size, self.in_features, 1, device=x.device), split_prob], dim=-1)
        right = torch.cat([split_prob, torch.zeros(batch_size, self.in_features, 1, device=x.device)], dim=-1)
        bin_probs = left * (1.0 - right)

        mu = bin_probs[:, 0, :]
        for d in range(1, self.in_features):
            next_prob = bin_probs[:, d, :]
            mu = torch.bmm(mu.unsqueeze(-1), next_prob.unsqueeze(1)).view(batch_size, -1)

        return torch.matmul(mu, self.leaf_weights)


class DecisionTree(nn.Module):
    """Oblique soft decision tree for DNDF (Kontschieder et al., 2015)."""
    def __init__(self, in_features: int, depth: int = 4, num_classes: int = 1):
        super().__init__()
        self.depth = depth
        self.num_internals = (2 ** depth) - 1
        self.num_leaves = 2 ** depth

        self.decision_layer = nn.Linear(in_features, self.num_internals)
        self.leaf_weights = nn.Parameter(torch.randn(self.num_leaves, num_classes) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        d_nodes = torch.sigmoid(self.decision_layer(x))

        mu = torch.ones(batch_size, 1, device=x.device)
        for d in range(self.depth):
            current_nodes = d_nodes[:, (2**d) - 1 : (2**(d + 1)) - 1]
            left_mu = mu * current_nodes
            right_mu = mu * (1.0 - current_nodes)
            mu = torch.stack([left_mu, right_mu], dim=-1).view(batch_size, -1)

        return torch.matmul(mu, self.leaf_weights)


class DeepNeuralDecisionForest(nn.Module):
    """DNDF: Differentiable ensemble of soft decision trees."""
    def __init__(self, in_features: int, num_trees: int = 10, depth: int = 4, num_classes: int = 1):
        super().__init__()
        self.trees = nn.ModuleList([
            DecisionTree(in_features=in_features, depth=depth, num_classes=num_classes)
            for _ in range(num_trees)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tree_outputs = [tree(x) for tree in self.trees]
        return torch.mean(torch.stack(tree_outputs, dim=0), dim=0)