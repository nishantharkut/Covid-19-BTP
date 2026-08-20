import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.models.deep_trees import DeepNeuralDecisionTree, DeepNeuralDecisionForest

def compute_bootstrap_ci(y_true, y_pred, n_bootstraps=1000, alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    aurocs, auprcs = [], []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        aurocs.append(roc_auc_score(y_true[idx], y_pred[idx]))
        auprcs.append(average_precision_score(y_true[idx], y_pred[idx]))
    
    return {
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_ci_low": float(np.percentile(aurocs, 100 * (alpha / 2))),
        "auroc_ci_high": float(np.percentile(aurocs, 100 * (1 - alpha / 2))),
        "auprc_mean": float(np.mean(auprcs)),
        "auprc_ci_low": float(np.percentile(auprcs, 100 * (alpha / 2))),
        "auprc_ci_high": float(np.percentile(auprcs, 100 * (1 - alpha / 2)))
    }

def train_eval_loop(model, X_train, y_train, X_val, y_val, lr=0.005, epochs=80, weight_decay=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    pos_weight = torch.tensor([(len(y_train) - sum(y_train)) / max(1, sum(y_train))], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)

    best_val_loss = float('inf')
    best_probs = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, torch.tensor(y_val, dtype=torch.float32, device=device).unsqueeze(1)).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_probs = torch.sigmoid(val_logits).cpu().numpy().ravel()

    return best_probs if best_probs is not None else torch.sigmoid(val_logits).cpu().numpy().ravel()

def run_experiment_pipeline(X, y, k_dndt=8, k_dndf=32, n_splits=5, dndt_epochs=60, dndf_epochs=80):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {"DNDT": {"y_true": [], "y_pred": []}, "DNDF": {"y_true": [], "y_pred": []}}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # 1. DNDT Branch
        fs_dndt = SelectKBest(f_classif, k=min(k_dndt, X.shape[1]))
        X_tr_dndt = fs_dndt.fit_transform(X_train_scaled, y_train)
        X_va_dndt = fs_dndt.transform(X_val_scaled)
        
        dndt_model = DeepNeuralDecisionTree(in_features=X_tr_dndt.shape[1], num_cutpoints=1, temperature=1.0)
        dndt_preds = train_eval_loop(dndt_model, X_tr_dndt, y_train, X_va_dndt, y_val, lr=0.01, epochs=dndt_epochs)
        
        results["DNDT"]["y_true"].extend(y_val)
        results["DNDT"]["y_pred"].extend(dndt_preds)

        # 2. DNDF Branch
        fs_dndf = SelectKBest(f_classif, k=min(k_dndf, X.shape[1]))
        X_tr_dndf = fs_dndf.fit_transform(X_train_scaled, y_train)
        X_va_dndf = fs_dndf.transform(X_val_scaled)

        dndf_model = DeepNeuralDecisionForest(in_features=X_tr_dndf.shape[1], num_trees=12, depth=4)
        dndf_preds = train_eval_loop(dndf_model, X_tr_dndf, y_train, X_va_dndf, y_val, lr=0.003, epochs=dndf_epochs)
        
        results["DNDF"]["y_true"].extend(y_val)
        results["DNDF"]["y_pred"].extend(dndf_preds)

    summary_rows = []
    for model_name in ["DNDT", "DNDF"]:
        y_t = np.array(results[model_name]["y_true"])
        y_p = np.array(results[model_name]["y_pred"])
        metrics = compute_bootstrap_ci(y_t, y_p)
        metrics["model"] = model_name
        summary_rows.append(metrics)
        print(f"[{model_name}] AUROC: {metrics['auroc_mean']:.3f} [{metrics['auroc_ci_low']:.3f}, {metrics['auroc_ci_high']:.3f}] | "
              f"AUPRC: {metrics['auprc_mean']:.3f} [{metrics['auprc_ci_low']:.3f}, {metrics['auprc_ci_high']:.3f}]")

    return pd.DataFrame(summary_rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run quick dry-run on synthetic data")
    parser.add_argument("--output_dir", type=str, default="data/outputs/metrics")
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY-RUN] Executing synthetic pipeline check...")
        X_dummy = np.random.randn(150, 64).astype(np.float32)
        y_dummy = np.random.binomial(1, 0.35, size=150).astype(np.int64)
        df_res = run_experiment_pipeline(X_dummy, y_dummy, k_dndt=6, k_dndf=20, n_splits=3, dndt_epochs=10, dndf_epochs=10)
        os.makedirs(args.output_dir, exist_ok=True)
        out_csv = os.path.join(args.output_dir, "dndt_dndf_dryrun_metrics.csv")
        df_res.to_csv(out_csv, index=False)
        print(f"[DRY-RUN COMPLETE] Saved sample metrics to {out_csv}")