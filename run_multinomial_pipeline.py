"""Multinomial xScore pipeline: train -> evaluate -> recompute GDS -> export."""
from pathlib import Path
import pandas as pd
import numpy as np
from src.data import (
    load_play_by_play, filter_offensive_snaps, add_drive_outcome_target,
    engineer_features, split_by_season, FEATURE_COLUMNS, OUTCOME_CLASSES,
    OUTCOME_TO_INDEX, normalize_drive_start_transition, compute_st_baselines,
    compute_st_value, compute_playoff_outcomes,
)
from src.model import (
    train_calibrated_multinomial, predict_multinomial, evaluate_multinomial,
    compute_xep_lookup_table, compute_xep, compute_drive_xvoa_ep,
    compute_gds_from_ep, save_model,
)
import joblib


def main():
    print("=" * 70)
    print("Multinomial xScore Pipeline (4-class: TD/FG/Turnover/Punt)")
    print("=" * 70)

    # 1. Data Preparation
    print("\n[1/7] Loading and preparing data...")
    raw_df = load_play_by_play(list(range(2018, 2026)))
    base_df = filter_offensive_snaps(raw_df)
    base_df = add_drive_outcome_target(base_df)
    base_df = engineer_features(base_df)
    base_df = base_df.dropna(subset=FEATURE_COLUMNS + ["drive_outcome"])

    splits = split_by_season(base_df)
    train = splits["train"]
    test = splits["test"]
    print(f"  Training: {len(train):,} plays (2018-2024)")
    print(f"  Test: {len(test):,} plays (2025)")

    print("\n  Class distribution (train):")
    for cls_name in OUTCOME_CLASSES:
        idx = OUTCOME_TO_INDEX[cls_name]
        count = (train["drive_outcome"] == idx).sum()
        print(f"    {cls_name}: {count:,} ({count/len(train):.1%})")

    X_train = train[FEATURE_COLUMNS]
    y_train = train["drive_outcome"]
    X_test = test[FEATURE_COLUMNS]
    y_test = test["drive_outcome"]

    # 2. Model Training
    print("\n[2/7] Training multinomial XGBoost with isotonic calibration...")
    model, calibrators = train_calibrated_multinomial(X_train, y_train, cv_folds=5)

    Path("models").mkdir(exist_ok=True)
    save_model(model, "models/xscore_multinomial.json")
    joblib.dump(calibrators, "models/xscore_multinomial_calibrators.pkl")
    print("  Model saved to models/xscore_multinomial.json")
    print("  Calibrators saved to models/xscore_multinomial_calibrators.pkl")

    # 3. Evaluation
    print("\n[3/7] Evaluating model...")
    metrics = evaluate_multinomial(model, X_test, y_test, calibrators=calibrators)
    print(f"  Multiclass Brier Score: {metrics['multiclass_brier']:.4f}")
    print(f"\n  Per-class AUC-ROC:")
    for cls_name, auc in metrics["per_class_auc"].items():
        print(f"    {cls_name}: {auc:.4f}")

    # 4. Build xEP Lookup Table
    print("\n[4/7] Building opponent expected-points lookup table...")
    train_drives = train.groupby(["game_id", "drive"]).first().reset_index()
    train_drives["actual_points"] = train_drives["drive_outcome"].map(
        {OUTCOME_TO_INDEX["td"]: 7, OUTCOME_TO_INDEX["fg"]: 3,
         OUTCOME_TO_INDEX["turnover"]: 0, OUTCOME_TO_INDEX["punt_other"]: 0}
    )
    # Flip yardline for opponent perspective
    opp_start_data = train_drives[["yardline_100", "actual_points"]].copy()
    opp_start_data = opp_start_data.rename(columns={"yardline_100": "start_yardline_100"})
    opp_start_data["start_yardline_100"] = 100 - opp_start_data["start_yardline_100"]
    opp_ep_lookup = compute_xep_lookup_table(opp_start_data)
    print(f"  Lookup table covers {len(opp_ep_lookup)} yardline positions")

    # 5. Recompute GDS
    print("\n[5/7] Recomputing GDS for all 2018-2024 drives...")
    reg_df = base_df[base_df["season_type"] == "REG"].copy() if "season_type" in base_df.columns else base_df.copy()

    drive_first = reg_df.groupby(["game_id", "posteam", "drive"]).first().reset_index()
    drive_probs = predict_multinomial(model, drive_first[FEATURE_COLUMNS], calibrators=calibrators)

    turnover_yardlines = 100 - drive_first["yardline_100"].values
    drive_first["xep"] = compute_xep(drive_probs, turnover_yardlines, opp_ep_lookup)

    drive_first["actual_points"] = drive_first["drive_outcome"].map(
        {OUTCOME_TO_INDEX["td"]: 7, OUTCOME_TO_INDEX["fg"]: 3,
         OUTCOME_TO_INDEX["turnover"]: 0, OUTCOME_TO_INDEX["punt_other"]: 0}
    )

    drive_xvoa = compute_drive_xvoa_ep(drive_first[["game_id", "posteam", "drive", "actual_points", "xep"]])

    # ST value
    reg_df = normalize_drive_start_transition(reg_df)
    all_probs = predict_multinomial(model, reg_df[FEATURE_COLUMNS], calibrators=calibrators)
    reg_df["xscore"] = all_probs[:, 1]
    st_baselines = compute_st_baselines(reg_df)
    st_value_df = compute_st_value(reg_df, st_baselines)

    game_gds = compute_gds_from_ep(drive_xvoa, st_value_df)
    print(f"  Computed GDS for {len(game_gds)} team-games")

    # 6. Season-Level Aggregation
    print("\n[6/7] Aggregating season-level GDS...")
    season_map = reg_df.groupby("game_id")["season"].first().to_dict()
    game_gds["season"] = game_gds["game_id"].map(season_map)

    def _get_opponent(row, gds_df):
        game_rows = gds_df[gds_df["game_id"] == row["game_id"]]
        opp = game_rows[game_rows["posteam"] != row["posteam"]]
        return opp["posteam"].iloc[0] if len(opp) > 0 else None

    game_gds["opponent"] = game_gds.apply(lambda r: _get_opponent(r, game_gds), axis=1)

    season_gds = game_gds.groupby(["season", "posteam"]).agg(
        games=("game_id", "count"),
        total_gds=("gds", "sum"),
        off_xvoa=("offensive_xvoa", "sum"),
        def_xvoa=("defensive_xvoa", "sum"),
        st_val=("st_value", "sum"),
    ).reset_index().rename(columns={"posteam": "team"})
    season_gds["gds_per_game"] = season_gds["total_gds"] / season_gds["games"]
    season_gds["off_xvoa_per_game"] = season_gds["off_xvoa"] / season_gds["games"]
    season_gds["def_xvoa_per_game"] = season_gds["def_xvoa"] / season_gds["games"]
    print(f"  {len(season_gds)} team-seasons computed")

    # 7. Export
    print("\n[7/7] Exporting results...")
    Path("output").mkdir(exist_ok=True)

    playoff_df = compute_playoff_outcomes(raw_df)
    merged = season_gds.merge(playoff_df, on=["season", "team"], how="left")
    merged["made_playoffs"] = merged["made_playoffs"].fillna(False)
    merged["playoff_wins"] = merged["playoff_wins"].fillna(0).astype(int)

    merged.to_csv("output/multinomial_gds_data.csv", index=False)
    game_gds.to_csv("output/multinomial_game_gds.csv", index=False)
    print("  Saved: output/multinomial_gds_data.csv")
    print("  Saved: output/multinomial_game_gds.csv")

    print(f"\n  Off xVOA/game range: [{merged['off_xvoa_per_game'].min():.2f}, {merged['off_xvoa_per_game'].max():.2f}]")
    print(f"  Def xVOA/game range: [{merged['def_xvoa_per_game'].min():.2f}, {merged['def_xvoa_per_game'].max():.2f}]")

    print("\n" + "=" * 70)
    print("Multinomial pipeline complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
