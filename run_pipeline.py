"""xScore end-to-end pipeline: load -> train -> evaluate -> visualize."""
from pathlib import Path
from src.data import (
    load_play_by_play, filter_offensive_snaps, add_drive_touchdown_target,
    engineer_features, engineer_phase2_rolling_features, split_by_season,
    FEATURE_COLUMNS, PHASE2_FEATURE_COLUMNS, TARGET_COLUMN,
)
from src.model import (
    train_xscore_model, predict_xscore, evaluate_model,
    check_intuition, save_model, compute_game_xscore,
    train_calibrated_model, save_calibrator,
)
from src.xdecision import build_ep_table, build_fg_pct_table, xdecision
from src.viz import plot_field_heatmap, plot_calibration, plot_shap_summary
import nfl_data_py as nfl
import pandas as pd


def main():
    print("=" * 60)
    print("xScore Pipeline")
    print("=" * 60)

    # 1. Data preparation (shared load for Phase 1 and Phase 2)
    print("\n[1/6] Loading and preparing data...")
    raw_df = load_play_by_play(list(range(2014, 2026)))
    base_df = filter_offensive_snaps(raw_df)
    base_df = add_drive_touchdown_target(base_df)
    base_df = engineer_features(base_df)

    # Phase 1 split
    p1_df = base_df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    splits = split_by_season(p1_df)
    train = splits["train"]
    test = splits["test"]
    print(f"  Training: {len(train):,} plays (2018-2024)")
    print(f"  Test: {len(test):,} plays (2025)")

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]
    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]

    # 2. Model training
    print("\n[2/6] Training XGBoost model...")
    model = train_xscore_model(X_train, y_train)
    save_model(model)
    print("  Model saved to models/xscore_v1.json")

    # 3. Evaluation
    print("\n[3/6] Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    print(f"  Brier Score: {metrics['brier_score']:.4f}")
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")

    intuition = check_intuition(model)
    print("\n  Intuition checks:")
    print(f"    1st & Goal from 1: {intuition['1st_goal_from_1']:.3f} (expect ~0.85)")
    print(f"    1st & Goal from 5: {intuition['1st_goal_from_5']:.3f} (expect ~0.60)")
    print(f"    1st & 10 own 30:   {intuition['1st_10_own_30']:.3f} (expect ~0.08)")
    print(f"    4th & 22 midfield: {intuition['4th_22_midfield']:.3f} (expect ~0.03)")

    # 4. Predictions and aggregation
    print("\n[4/6] Computing xScore for all plays...")
    test = test.copy()
    test["xscore"] = predict_xscore(model, X_test)
    game_xscores = compute_game_xscore(test[["game_id", "posteam", "drive", "xscore"]])
    print(f"  Computed game xScore for {len(game_xscores)} team-games")

    # 5. xDecision setup
    print("\n[5/6] Building xDecision tables...")
    train = train.copy()
    train["xscore"] = predict_xscore(model, X_train)
    ep_table = build_ep_table(train[["yardline_100", "xscore"]])
    raw_pbp = nfl.import_pbp_data(list(range(2018, 2025)))
    fg_pct = build_fg_pct_table(raw_pbp)
    sample = xdecision(yardline_100=35, ydstogo=3, ep_table=ep_table, fg_pct=fg_pct)
    print(f"  Sample xDecision (4th & 3 from opp 35):")
    print(f"    Go: {sample['go_ep']:.3f} EP | Kick: {sample['kick_ep']:.3f} EP | Punt: {sample['punt_ep']:.3f} EP")
    print(f"    Recommendation: {sample['recommendation'].upper()}")

    # 6. Visualizations
    print("\n[6/6] Generating visualizations...")
    Path("outputs").mkdir(exist_ok=True)
    plot_field_heatmap(test[["down", "yardline_100", "xscore"]], "outputs/heatmap.png")
    plot_calibration(metrics, "outputs/calibration.png")
    plot_shap_summary(model, X_test.sample(min(1000, len(X_test)), random_state=42), "outputs/shap.png")
    print("  Saved: outputs/heatmap.png")
    print("  Saved: outputs/calibration.png")
    print("  Saved: outputs/shap.png")

    # Phase 2: Enhanced model
    print("\n" + "=" * 60)
    print("Phase 2: Enhanced xScore (Rolling EPA + Momentum + Calibration)")
    print("=" * 60)

    print("\n[P2-1/4] Engineering Phase 2 features (rolling EPA + momentum)...")
    p2_df = engineer_phase2_rolling_features(base_df.copy())
    p2_df = p2_df.dropna(subset=PHASE2_FEATURE_COLUMNS + [TARGET_COLUMN])
    p2_splits = split_by_season(p2_df)
    p2_train = p2_splits["train"]
    p2_test = p2_splits["test"]
    print(f"  Training: {len(p2_train):,} plays")
    print(f"  Test: {len(p2_test):,} plays")

    X_p2_train = p2_train[PHASE2_FEATURE_COLUMNS]
    y_p2_train = p2_train[TARGET_COLUMN]
    X_p2_test = p2_test[PHASE2_FEATURE_COLUMNS]
    y_p2_test = p2_test[TARGET_COLUMN]

    # Phase 2 without calibration
    print("\n[P2-2/4] Training Phase 2 model (no calibration)...")
    p2_model_raw = train_xscore_model(X_p2_train, y_p2_train)
    p2_metrics_raw = evaluate_model(p2_model_raw, X_p2_test, y_p2_test)
    print(f"  Brier Score: {p2_metrics_raw['brier_score']:.4f}")
    print(f"  AUC-ROC: {p2_metrics_raw['auc_roc']:.4f}")

    # Phase 2 with calibration
    print("\n[P2-3/4] Training Phase 2 model (with isotonic calibration)...")
    p2_model, p2_calibrator = train_calibrated_model(X_p2_train, y_p2_train)
    save_model(p2_model, "models/xscore_v2.json")
    save_calibrator(p2_calibrator)
    p2_preds_cal = predict_xscore(p2_model, X_p2_test, calibrator=p2_calibrator)
    from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402
    p2_brier_cal = brier_score_loss(y_p2_test, p2_preds_cal)
    p2_auc_cal = roc_auc_score(y_p2_test, p2_preds_cal)
    print(f"  Brier Score: {p2_brier_cal:.4f}")
    print(f"  AUC-ROC: {p2_auc_cal:.4f}")

    # A/B Comparison
    print("\n[P2-4/4] A/B Comparison")
    print(f"  {'Metric':<15} {'Phase 1':<12} {'P2 (raw)':<12} {'P2 (cal.)':<12}")
    print(f"  {'-'*51}")
    print(f"  {'Brier Score':<15} {metrics['brier_score']:<12.4f} {p2_metrics_raw['brier_score']:<12.4f} {p2_brier_cal:<12.4f}")
    print(f"  {'AUC-ROC':<15} {metrics['auc_roc']:<12.4f} {p2_metrics_raw['auc_roc']:<12.4f} {p2_auc_cal:<12.4f}")

    # Phase 2 intuition checks
    p2_scenarios = pd.DataFrame([
        {"down": 1, "ydstogo": 1, "yardline_100": 1, "score_diff": 0,
         "half_seconds_remaining": 900, "goal_to_go": 1, "red_zone": 1,
         "rolling_offense_epa": 0.05, "rolling_defense_epa": 0.0,
         "momentum_epa": 0.1, "is_home": 1},
        {"down": 1, "ydstogo": 5, "yardline_100": 5, "score_diff": 0,
         "half_seconds_remaining": 900, "goal_to_go": 1, "red_zone": 1,
         "rolling_offense_epa": 0.05, "rolling_defense_epa": 0.0,
         "momentum_epa": 0.1, "is_home": 1},
        {"down": 1, "ydstogo": 10, "yardline_100": 70, "score_diff": 0,
         "half_seconds_remaining": 900, "goal_to_go": 0, "red_zone": 0,
         "rolling_offense_epa": 0.05, "rolling_defense_epa": 0.0,
         "momentum_epa": 0.0, "is_home": 1},
        {"down": 4, "ydstogo": 22, "yardline_100": 50, "score_diff": 0,
         "half_seconds_remaining": 900, "goal_to_go": 0, "red_zone": 0,
         "rolling_offense_epa": 0.0, "rolling_defense_epa": 0.0,
         "momentum_epa": -0.1, "is_home": 0},
    ])
    p2_intuition = predict_xscore(p2_model, p2_scenarios, calibrator=p2_calibrator)
    print("\n  Phase 2 Intuition checks (calibrated):")
    print(f"    1st & Goal from 1: {p2_intuition[0]:.3f} (expect ~0.85-0.92)")
    print(f"    1st & Goal from 5: {p2_intuition[1]:.3f} (expect ~0.55-0.65)")
    print(f"    1st & 10 own 30:   {p2_intuition[2]:.3f} (expect ~0.20-0.25)")
    print(f"    4th & 22 midfield: {p2_intuition[3]:.3f} (expect ~0.03-0.05)")

    # Phase 2 SHAP
    print("\n  Generating Phase 2 SHAP plot...")
    plot_shap_summary(
        p2_model,
        X_p2_test.sample(min(1000, len(X_p2_test)), random_state=42),
        "outputs/shap_phase2.png",
    )
    print("  Saved: outputs/shap_phase2.png")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
