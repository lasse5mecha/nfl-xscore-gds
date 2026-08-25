"""Compute drive-clustered bootstrap 95% CIs for all key xScore metrics.

Resamples drives (not plays) with replacement to respect within-drive
outcome-label dependence. 5,000 iterations, percentile method.
"""
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
from src.data import (
    load_play_by_play, filter_offensive_snaps, add_drive_outcome_target,
    engineer_features, split_by_season, FEATURE_COLUMNS, OUTCOME_CLASSES,
    OUTCOME_TO_INDEX,
)
from src.model import predict_multinomial, load_model

N_BOOTSTRAP = 5000
SEED = 42
ALPHA = 0.05


def compute_multiclass_brier(probs, y_arr, n_classes=4):
    brier_per_class = []
    for cls in range(n_classes):
        binary_y = (y_arr == cls).astype(int)
        brier_per_class.append(np.mean((probs[:, cls] - binary_y) ** 2))
    return np.mean(brier_per_class)


def compute_naive_brier(y_arr, train_y, n_classes=4):
    marginals = np.array([np.mean(train_y == cls) for cls in range(n_classes)])
    brier_per_class = []
    for cls in range(n_classes):
        binary_y = (y_arr == cls).astype(int)
        brier_per_class.append(np.mean((marginals[cls] - binary_y) ** 2))
    return np.mean(brier_per_class)


def compute_ece(probs, y_arr, n_classes=4, n_bins=10):
    ece_per_class = []
    for cls in range(n_classes):
        binary_y = (y_arr == cls).astype(int)
        p = probs[:, cls]
        sorted_idx = np.argsort(p)
        bins = np.array_split(sorted_idx, n_bins)
        abs_diffs = []
        for b in bins:
            if len(b) == 0:
                continue
            mean_pred = p[b].mean()
            mean_obs = binary_y[b].mean()
            abs_diffs.append(abs(mean_pred - mean_obs))
        ece_per_class.append(np.mean(abs_diffs))
    return np.mean(ece_per_class)


def compute_per_class_auc(probs, y_arr, n_classes=4):
    aucs = {}
    for cls in range(n_classes):
        binary_y = (y_arr == cls).astype(int)
        if binary_y.sum() > 0 and binary_y.sum() < len(binary_y):
            aucs[cls] = roc_auc_score(binary_y, probs[:, cls])
        else:
            aucs[cls] = np.nan
    return aucs


def run_bootstrap(drive_ids, drive_to_indices, probs, y_arr, train_y,
                  lr_probs, n_bootstrap, rng):
    n_drives = len(drive_ids)
    n_classes = 4

    results = {
        "brier": np.zeros(n_bootstrap),
        "bss_naive": np.zeros(n_bootstrap),
        "bss_lr": np.zeros(n_bootstrap),
        "ece": np.zeros(n_bootstrap),
        "auc_0": np.zeros(n_bootstrap),
        "auc_1": np.zeros(n_bootstrap),
        "auc_2": np.zeros(n_bootstrap),
        "auc_3": np.zeros(n_bootstrap),
    }

    for i in range(n_bootstrap):
        if (i + 1) % 500 == 0:
            print(f"  Bootstrap iteration {i+1}/{n_bootstrap}...")

        sampled_drives = rng.choice(drive_ids, size=n_drives, replace=True)
        idx = np.concatenate([drive_to_indices[d] for d in sampled_drives])

        p_boot = probs[idx]
        y_boot = y_arr[idx]

        bs = compute_multiclass_brier(p_boot, y_boot, n_classes)
        results["brier"][i] = bs

        bs_naive = compute_naive_brier(y_boot, train_y, n_classes)
        results["bss_naive"][i] = 1 - bs / bs_naive

        bs_lr = compute_multiclass_brier(lr_probs[idx], y_boot, n_classes)
        results["bss_lr"][i] = 1 - bs / bs_lr

        results["ece"][i] = compute_ece(p_boot, y_boot, n_classes)

        aucs = compute_per_class_auc(p_boot, y_boot, n_classes)
        for cls in range(n_classes):
            results[f"auc_{cls}"][i] = aucs[cls]

    return results


def run_ood_bootstrap(drive_ids, drive_to_indices, probs, y_arr,
                      n_bootstrap, rng):
    n_drives = len(drive_ids)
    results = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        if (i + 1) % 500 == 0:
            print(f"  OOD bootstrap iteration {i+1}/{n_bootstrap}...")

        sampled_drives = rng.choice(drive_ids, size=n_drives, replace=True)
        idx = np.concatenate([drive_to_indices[d] for d in sampled_drives])

        results[i] = compute_multiclass_brier(probs[idx], y_arr[idx])

    return results


def main():
    print("=" * 70)
    print("Bootstrap Confidence Intervals for xScore Metrics")
    print(f"  Iterations: {N_BOOTSTRAP}, Seed: {SEED}, Alpha: {ALPHA}")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading and preparing data...")
    raw_df = load_play_by_play(list(range(2014, 2026)))
    base_df = filter_offensive_snaps(raw_df)
    base_df = add_drive_outcome_target(base_df)
    base_df = engineer_features(base_df)
    base_df = base_df.dropna(subset=FEATURE_COLUMNS + ["drive_outcome"])

    splits = split_by_season(base_df)
    train = splits["train"]
    test = splits["test"]
    ood = splits["ood"]
    print(f"  Train: {len(train):,} plays")
    print(f"  Test:  {len(test):,} plays, {test.groupby(['game_id','drive']).ngroup().nunique():,} drives")
    print(f"  OOD:   {len(ood):,} plays, {ood.groupby(['game_id','drive']).ngroup().nunique():,} drives")

    # Load model
    print("\n[2/5] Loading model and generating predictions...")
    model = load_model("models/xscore_multinomial.json")
    calibrators = joblib.load("models/xscore_multinomial_calibrators.pkl")

    X_train = train[FEATURE_COLUMNS]
    y_train = train["drive_outcome"].values
    X_test = test[FEATURE_COLUMNS]
    y_test = test["drive_outcome"].values
    X_ood = ood[FEATURE_COLUMNS]
    y_ood = ood["drive_outcome"].values

    probs_test = predict_multinomial(model, X_test, calibrators=calibrators)
    probs_ood = predict_multinomial(model, X_ood, calibrators=calibrators)

    # Train logistic regression for BSS comparison
    print("  Training multinomial logistic regression baseline...")
    lr = LogisticRegression(max_iter=5000, solver="lbfgs", random_state=42)
    lr.fit(X_train, y_train)
    lr_probs_test = lr.predict_proba(X_test)

    # Point estimates
    print("\n[3/5] Computing point estimates...")
    bs_point = compute_multiclass_brier(probs_test, y_test)
    bs_naive_point = compute_naive_brier(y_test, y_train)
    bs_lr_point = compute_multiclass_brier(lr_probs_test, y_test)
    bss_naive_point = 1 - bs_point / bs_naive_point
    bss_lr_point = 1 - bs_point / bs_lr_point
    ece_point = compute_ece(probs_test, y_test)
    aucs_point = compute_per_class_auc(probs_test, y_test)
    bs_ood_point = compute_multiclass_brier(probs_ood, y_ood)

    print(f"  Brier Score:       {bs_point:.4f}")
    print(f"  Naive Brier:       {bs_naive_point:.4f}")
    print(f"  LR Brier:          {bs_lr_point:.4f}")
    print(f"  BSS vs Naive:      {bss_naive_point:.4f} ({bss_naive_point*100:.1f}%)")
    print(f"  BSS vs LR:         {bss_lr_point:.4f} ({bss_lr_point*100:.1f}%)")
    print(f"  ECE:               {ece_point:.4f}")
    print(f"  AUC punt_other:    {aucs_point[0]:.4f}")
    print(f"  AUC td:            {aucs_point[1]:.4f}")
    print(f"  AUC fg:            {aucs_point[2]:.4f}")
    print(f"  AUC turnover:      {aucs_point[3]:.4f}")
    print(f"  OOD Brier:         {bs_ood_point:.4f}")

    # Build drive cluster mappings
    print("\n[4/5] Running drive-clustered bootstrap on TEST set...")
    test_drive_col = test["game_id"].astype(str) + "_" + test["drive"].astype(str)
    test_drive_ids = test_drive_col.unique()
    test_drive_to_idx = {d: np.where(test_drive_col.values == d)[0] for d in test_drive_ids}
    print(f"  {len(test_drive_ids):,} drives, {len(y_test):,} plays")

    rng = np.random.default_rng(SEED)
    test_results = run_bootstrap(
        test_drive_ids, test_drive_to_idx, probs_test, y_test, y_train,
        lr_probs_test, N_BOOTSTRAP, rng
    )

    # OOD bootstrap
    print("\n[5/5] Running drive-clustered bootstrap on OOD set...")
    ood_drive_col = ood["game_id"].astype(str) + "_" + ood["drive"].astype(str)
    ood_drive_ids = ood_drive_col.unique()
    ood_drive_to_idx = {d: np.where(ood_drive_col.values == d)[0] for d in ood_drive_ids}
    print(f"  {len(ood_drive_ids):,} drives, {len(y_ood):,} plays")

    ood_brier_results = run_ood_bootstrap(
        ood_drive_ids, ood_drive_to_idx, probs_ood, y_ood, N_BOOTSTRAP, rng
    )

    # Compute CIs (percentile method)
    lo = ALPHA / 2 * 100
    hi = (1 - ALPHA / 2) * 100

    print("\n" + "=" * 70)
    print("RESULTS: 95% Confidence Intervals (Percentile Method)")
    print("=" * 70)

    def ci_str(point, samples):
        return f"{point:.4f}, 95% CI [{np.percentile(samples, lo):.4f}, {np.percentile(samples, hi):.4f}]"

    print(f"\n  Multiclass Brier Score (test):  {ci_str(bs_point, test_results['brier'])}")
    print(f"  BSS vs Naive:                   {ci_str(bss_naive_point, test_results['bss_naive'])}")
    print(f"  BSS vs Logistic Regression:     {ci_str(bss_lr_point, test_results['bss_lr'])}")
    print(f"  Mean ECE:                       {ci_str(ece_point, test_results['ece'])}")

    print(f"\n  Per-class AUC-ROC:")
    class_names = OUTCOME_CLASSES
    for cls in range(4):
        print(f"    {class_names[cls]:12s}: {ci_str(aucs_point[cls], test_results[f'auc_{cls}'])}")

    print(f"\n  OOD Brier Score (2014-2017):    {ci_str(bs_ood_point, ood_brier_results)}")

    # Also print in paper-ready format (paper order: TD, FG, Turnover, Punt/Other)
    print("\n" + "=" * 70)
    print("PAPER-READY FORMAT (paper class order: TD, FG, Turnover, Punt/Other)")
    print("=" * 70)

    paper_order = [1, 2, 3, 0]  # td, fg, turnover, punt_other
    paper_names = ["TD", "FG", "Turnover", "Punt/Other"]

    brier_lo = np.percentile(test_results["brier"], lo)
    brier_hi = np.percentile(test_results["brier"], hi)
    print(f"\n  Brier: {bs_point:.4f}, 95% CI [{brier_lo:.4f}, {brier_hi:.4f}]")

    bss_n_lo = np.percentile(test_results["bss_naive"], lo)
    bss_n_hi = np.percentile(test_results["bss_naive"], hi)
    print(f"  BSS vs Naive: {bss_naive_point*100:.1f}%, 95% CI [{bss_n_lo*100:.1f}%, {bss_n_hi*100:.1f}%]")

    bss_lr_lo = np.percentile(test_results["bss_lr"], lo)
    bss_lr_hi = np.percentile(test_results["bss_lr"], hi)
    print(f"  BSS vs LR: {bss_lr_point*100:.1f}%, 95% CI [{bss_lr_lo*100:.1f}%, {bss_lr_hi*100:.1f}%]")

    ece_lo = np.percentile(test_results["ece"], lo)
    ece_hi = np.percentile(test_results["ece"], hi)
    print(f"  ECE: {ece_point:.3f}, 95% CI [{ece_lo:.3f}, {ece_hi:.3f}]")

    print(f"\n  Per-class AUC-ROC (paper order):")
    for i, cls in enumerate(paper_order):
        auc_lo = np.percentile(test_results[f"auc_{cls}"], lo)
        auc_hi = np.percentile(test_results[f"auc_{cls}"], hi)
        print(f"    {paper_names[i]:12s}: {aucs_point[cls]:.3f}, 95% CI [{auc_lo:.3f}, {auc_hi:.3f}]")

    ood_lo = np.percentile(ood_brier_results, lo)
    ood_hi = np.percentile(ood_brier_results, hi)
    print(f"\n  OOD Brier: {bs_ood_point:.4f}, 95% CI [{ood_lo:.4f}, {ood_hi:.4f}]")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
