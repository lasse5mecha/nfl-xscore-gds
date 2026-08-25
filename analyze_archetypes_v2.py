"""GDS Archetype Analysis v2: Multinomial xEP + Opponent Adjustment + Mediation."""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from pathlib import Path
import joblib

from src.data import (
    load_play_by_play, filter_offensive_snaps, add_drive_outcome_target,
    engineer_features, FEATURE_COLUMNS, OUTCOME_TO_INDEX,
    normalize_drive_start_transition, compute_st_baselines, compute_st_value,
    compute_playoff_outcomes,
)
from src.model import (
    predict_multinomial, compute_xep, compute_xep_lookup_table,
    compute_drive_xvoa_ep, compute_gds_from_ep,
)
from src.robustness import (
    compute_opponent_adjustment, compute_field_position_mediation,
    run_controlled_regressions,
)


def load_trained_model():
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model("models/xscore_multinomial.json")
    calibrators = joblib.load("models/xscore_multinomial_calibrators.pkl")
    return model, calibrators


def compute_offense_share(row):
    total = abs(row["off_xvoa_per_game"]) + abs(row["def_xvoa_per_game"]) + 0.01
    return row["off_xvoa_per_game"] / total


def main():
    print("=" * 70)
    print("GDS Archetype Analysis v2 (Multinomial xEP Model)")
    print("=" * 70)

    model, calibrators = load_trained_model()

    # Data Processing
    print("\nLoading 2018-2025 data...")
    raw_df = load_play_by_play(list(range(2018, 2026)))
    reg_df = raw_df[raw_df["season_type"] == "REG"].copy()
    df = filter_offensive_snaps(reg_df)
    df = add_drive_outcome_target(df)
    df = engineer_features(df)
    df = df.dropna(subset=FEATURE_COLUMNS + ["drive_outcome"])
    print(f"  {len(df):,} regular season snaps")

    # Compute Drive-Level xEP
    print("  Computing drive-level expected points...")
    drive_first = df.groupby(["game_id", "posteam", "drive"]).first().reset_index()
    drive_probs = predict_multinomial(model, drive_first[FEATURE_COLUMNS], calibrators=calibrators)

    drive_first["actual_points"] = drive_first["drive_outcome"].map(
        {OUTCOME_TO_INDEX["td"]: 7, OUTCOME_TO_INDEX["fg"]: 3,
         OUTCOME_TO_INDEX["turnover"]: 0, OUTCOME_TO_INDEX["punt_other"]: 0}
    )
    opp_start_data = drive_first[["yardline_100", "actual_points"]].copy()
    opp_start_data = opp_start_data.rename(columns={"yardline_100": "start_yardline_100"})
    opp_start_data["start_yardline_100"] = 100 - opp_start_data["start_yardline_100"]
    opp_ep_lookup = compute_xep_lookup_table(opp_start_data)

    turnover_yardlines = 100 - drive_first["yardline_100"].values
    drive_first["xep"] = compute_xep(drive_probs, turnover_yardlines, opp_ep_lookup)

    drive_xvoa = compute_drive_xvoa_ep(
        drive_first[["game_id", "posteam", "drive", "actual_points", "xep"]]
    )

    # ST value
    df = normalize_drive_start_transition(df)
    all_probs = predict_multinomial(model, df[FEATURE_COLUMNS], calibrators=calibrators)
    df["xscore"] = all_probs[:, 1]
    st_baselines = compute_st_baselines(df)
    st_value_df = compute_st_value(df, st_baselines)

    # Game-level GDS
    game_gds = compute_gds_from_ep(drive_xvoa, st_value_df)

    # Season aggregation
    season_map = df.groupby("game_id")["season"].first().to_dict()
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

    playoff_df = compute_playoff_outcomes(raw_df)
    merged = season_gds.merge(playoff_df, on=["season", "team"], how="left")
    merged["made_playoffs"] = merged["made_playoffs"].fillna(False)
    merged["playoff_wins"] = merged["playoff_wins"].fillna(0).astype(int)
    merged["won_super_bowl"] = merged["won_super_bowl"].fillna(False)
    merged["win_pct"] = merged["reg_wins"] / merged["reg_games"]
    merged["offense_share"] = merged.apply(compute_offense_share, axis=1)
    print(f"  {len(merged)} team-seasons")

    # LAYER 1: Correlations
    print("\n" + "=" * 70)
    print("LAYER 1: Descriptive Analysis (Multinomial xEP Model)")
    print("=" * 70)

    r_off, p_off = stats.pearsonr(merged["off_xvoa_per_game"], merged["win_pct"])
    r_def, p_def = stats.pearsonr(merged["def_xvoa_per_game"], merged["win_pct"])
    print(f"\n  Off xVOA/game vs Win%: r={r_off:.3f} (p={p_off:.6f})")
    print(f"  Def xVOA/game vs Win%: r={r_def:.3f} (p={p_def:.6f})")
    if r_def != 0:
        print(f"  Variance ratio: {r_off**2/r_def**2:.1f}:1 (offense:defense)")

    playoff_teams = merged[merged["made_playoffs"]]
    rho, p_rho = stats.spearmanr(playoff_teams["offense_share"], playoff_teams["playoff_wins"])
    print(f"\n  Spearman (playoff teams): rho={rho:.3f} (p={p_rho:.4f}, n={len(playoff_teams)})")

    # LAYER 2: Statistical Tests
    print("\n" + "=" * 70)
    print("LAYER 2: Statistical Tests")
    print("=" * 70)

    # NOTE: Cohen's d is computed across ALL team-seasons (merged), not just
    # playoff qualifiers -- this matches the documented methodology (paper3:
    # "across all 224 observations") and is what captures both the selection
    # effect (making the playoffs) and the advancement effect (winning once
    # there). Restricting to playoff_teams here was a pre-existing bug that
    # silently understated the effect (verified: full-sample method on the
    # 2018-2024 subset reproduces the published d=0.672 almost exactly; the
    # playoff-teams-only version does not).
    off_dom = merged[merged["offense_share"] > 0.3]["playoff_wins"]
    def_dom = merged[merged["offense_share"] < -0.3]["playoff_wins"]
    if len(off_dom) > 0 and len(def_dom) > 0:
        pooled_std = np.sqrt(
            ((len(off_dom)-1)*off_dom.std()**2 + (len(def_dom)-1)*def_dom.std()**2)
            / (len(off_dom) + len(def_dom) - 2)
        )
        d = (off_dom.mean() - def_dom.mean()) / pooled_std if pooled_std > 0 else 0
        print(f"\n  Cohen's d (off-dom vs def-dom, all team-seasons): {d:.3f} (n_off={len(off_dom)}, n_def={len(def_dom)})")

    X_lr = merged[["off_xvoa_per_game", "def_xvoa_per_game"]].copy()
    X_lr = sm.add_constant(X_lr)
    y_lr = merged["won_super_bowl"].astype(int)
    try:
        logit = sm.Logit(y_lr, X_lr).fit(disp=0)
        print(f"\n  Logistic (P(SB win)):")
        print(f"    off_xvoa coef={logit.params['off_xvoa_per_game']:.3f} (p={logit.pvalues['off_xvoa_per_game']:.4f})")
        print(f"    def_xvoa coef={logit.params['def_xvoa_per_game']:.3f} (p={logit.pvalues['def_xvoa_per_game']:.4f})")
    except Exception as e:
        print(f"  Logistic regression failed: {e}")

    # LAYER 3: Opponent Adjustment
    print("\n" + "=" * 70)
    print("LAYER 3: Opponent Strength Adjustment (Leave-One-Out)")
    print("=" * 70)

    game_gds_for_opp = game_gds[["season", "posteam", "game_id", "opponent", "gds"]].copy()
    game_gds_for_opp = game_gds_for_opp.rename(columns={"posteam": "team"})
    opp_adj = compute_opponent_adjustment(game_gds_for_opp)
    merged = merged.merge(opp_adj, on=["season", "team"], how="left")
    merged["opp_avg_gds"] = merged["opp_avg_gds"].fillna(0)

    print(f"\n  Opponent strength range: [{merged['opp_avg_gds'].min():.2f}, {merged['opp_avg_gds'].max():.2f}]")

    reg_data = merged[["off_xvoa_per_game", "def_xvoa_per_game", "opp_avg_gds",
                       "made_playoffs", "playoff_wins"]].copy()
    reg_data["made_playoffs"] = reg_data["made_playoffs"].astype(int)
    reg_results = run_controlled_regressions(reg_data)

    print(f"\n  OLS (playoff_wins) -- Uncontrolled:")
    print(f"    off_xvoa: coef={reg_results['ols_uncontrolled']['off_coef']:.3f} (p={reg_results['ols_uncontrolled']['off_pvalue']:.4f})")
    print(f"    def_xvoa: coef={reg_results['ols_uncontrolled']['def_coef']:.3f} (p={reg_results['ols_uncontrolled']['def_pvalue']:.4f})")
    print(f"    R2={reg_results['ols_uncontrolled']['r_squared']:.3f}")

    print(f"\n  OLS (playoff_wins) -- With Opponent Adjustment:")
    print(f"    off_xvoa: coef={reg_results['ols_controlled']['off_coef']:.3f} (p={reg_results['ols_controlled']['off_pvalue']:.4f})")
    print(f"    def_xvoa: coef={reg_results['ols_controlled']['def_coef']:.3f} (p={reg_results['ols_controlled']['def_pvalue']:.4f})")
    print(f"    opp_avg_gds: coef={reg_results['ols_controlled']['opp_coef']:.3f} (p={reg_results['ols_controlled']['opp_pvalue']:.4f})")
    print(f"    R2={reg_results['ols_controlled']['r_squared']:.3f}")

    sig_held = reg_results['ols_controlled']['off_pvalue'] < 0.05
    print(f"\n  Verdict: Offense coefficient {'REMAINS SIGNIFICANT' if sig_held else 'LOST SIGNIFICANCE'} after opponent control")

    # LAYER 4: Field Position Mediation
    print("\n" + "=" * 70)
    print("LAYER 4: Field Position Mediation Analysis")
    print("=" * 70)

    avg_start = drive_first.groupby(["season", "posteam"])["yardline_100"].mean().reset_index()
    avg_start.columns = ["season", "team", "avg_start_yardline"]
    merged = merged.merge(avg_start, on=["season", "team"], how="left")

    mediation = compute_field_position_mediation(merged)
    print(f"\n  Path a (Def_xVOA -> Field Position): r={mediation['def_to_field_pos_r']:.3f} (p={mediation['def_to_field_pos_p']:.4f})")
    print(f"  Total effect (Off_xVOA -> Wins): r={mediation['off_xvoa_to_wins_r']:.3f}")
    print(f"  Partial (controlling field pos): r={mediation['off_xvoa_to_wins_partial_r']:.3f}")
    print(f"  Mediation percentage: {mediation['mediation_pct']:.1f}%")
    print(f"\n  Interpretation: {100-mediation['mediation_pct']:.0f}% of offensive value is independent of defense-generated field position")

    # Export
    print("\n" + "=" * 70)
    print("Exporting final results...")
    Path("output").mkdir(exist_ok=True)
    merged.to_csv("output/archetype_v2_data.csv", index=False)
    print("  Saved: output/archetype_v2_data.csv")

    # Summary
    print("\n" + "=" * 70)
    print("THESIS SUMMARY (Multinomial xEP Model)")
    print("=" * 70)
    print(f"  Offense explains {r_off**2:.1%} of win% variance")
    print(f"  Defense explains {r_def**2:.1%} of win% variance")
    if r_def != 0:
        print(f"  Ratio: {r_off**2/r_def**2:.1f}:1")
    print(f"  Offense-primacy holds after opponent adjustment: {'YES' if sig_held else 'NO'}")
    print(f"  Field position mediation: {mediation['mediation_pct']:.0f}% (spec threshold: <30%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
