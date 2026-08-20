import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from app.models.deep_trees import DeepNeuralDecisionTree, DeepNeuralDecisionForest

def test_dndt_dndf_contracts():
    batch_size = 8
    in_features_dndt = 6
    in_features_dndf = 32

    # 1. Test DNDT
    x_dndt = torch.randn(batch_size, in_features_dndt, requires_grad=True)
    dndt = DeepNeuralDecisionTree(in_features=in_features_dndt, num_cutpoints=1, num_classes=1)
    
    out_dndt = dndt(x_dndt)
    assert out_dndt.shape == (batch_size, 1), f"DNDT output shape mismatch: {out_dndt.shape}"
    
    loss_dndt = out_dndt.sum()
    loss_dndt.backward()
    
    for name, param in dndt.named_parameters():
        assert param.grad is not None, f"DNDT gradient missing for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in DNDT {name}"

    # 2. Test DNDF
    x_dndf = torch.randn(batch_size, in_features_dndf, requires_grad=True)
    dndf = DeepNeuralDecisionForest(in_features=in_features_dndf, num_trees=4, depth=3, num_classes=1)
    
    out_dndf = dndf(x_dndf)
    assert out_dndf.shape == (batch_size, 1), f"DNDF output shape mismatch: {out_dndf.shape}"
    
    loss_dndf = out_dndf.sum()
    loss_dndf.backward()
    
    for name, param in dndf.named_parameters():
        assert param.grad is not None, f"DNDF gradient missing for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in DNDF {name}"

    print("[SUCCESS] All shape, forward, and backward gradient assertions passed!")

if __name__ == "__main__":
    test_dndt_dndf_contracts()