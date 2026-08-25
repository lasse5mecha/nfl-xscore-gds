"""GDS Archetype Analysis: Team Builds vs Playoff Success (2018-2024).

Tests the thesis: 'Defense wins championships' is a myth. Offense-dominant
GDS profiles produce higher playoff advancement rates, but Super Bowl winners
tend to be offense-heavy WITH competent defense.
"""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from src.data import (
    load_play_by_play, filter_offensive_snaps, add_drive_touchdown_target,
    engineer_features, FEATURE_COLUMNS, TARGET_COLUMN,
    normalize_drive_start_transition, compute_st_baselines, compute_st_value,
    compute_playoff_outcomes,
)
from src.model import load_model, predict_xscore, compute_xvoa, compute_game_deserved_score


def compute_season_gds(df: pd.DataFrame, st_baselines: dict) -> pd.DataFrame:
    """Compute GDS decomposition per team per season from preprocessed data."""
    all_seasons = []

    for season in sorted(df["season"].unique()):
        season_df = df[df["season"] == season]
        game_xvoa = compute_xvoa(
            season_df[["game_id", "posteam", "drive", "xscore", "drive_td", "half_seconds_remaining"]]
        )
        st_value_df = compute_st_value(season_df, st_baselines)
        gds_df = compute_game_deserved_score(game_xvoa, st_value_df)

        season_summary = gds_df.groupby("posteam").agg(
            games=("game_id", "count"),
            total_gds=("gds", "sum"),
            off_xvoa=("offensive_xvoa", "sum"),
            def_xvoa=("defensive_xvoa", "sum"),
            st_val=("st_value", "sum"),
        ).reset_index()
        season_summary["season"] = season
        season_summary["gds_per_game"] = season_summary["total_gds"] / season_summary["games"]
        season_summary["off_xvoa_per_game"] = season_summary["off_xvoa"] / season_summary["games"]
        season_summary["def_xvoa_per_game"] = season_summary["def_xvoa"] / season_summary["games"]
        all_seasons.append(season_summary)

    return pd.concat(all_seasons, ignore_index=True).rename(columns={"posteam": "team"})


def compute_offense_share(row):
    """Offense share: -1 (pure defense) to +1 (pure offense)."""
    total = abs(row["off_xvoa_per_game"]) + abs(row["def_xvoa_per_game"]) + 0.01
    return row["off_xvoa_per_game"] / total


def main():
    print("=" * 70)
    print("GDS Archetype Analysis: Team Builds vs Playoff Success (2018-2024)")
    print("Thesis: 'Defense wins championships' — myth or reality?")
    print("=" * 70)

    model = load_model("models/xscore_v1.json")

    # === Load and process data ===
    print("\nLoading 2018-2024 data...")
    raw_df = load_play_by_play(list(range(2018, 2025)))
    print("  Processing regular season plays...")
    reg_df = raw_df[raw_df["season_type"] == "REG"].copy()
    df = filter_offensive_snaps(reg_df)
    df = add_drive_touchdown_target(df)
    df = engineer_features(df)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    df["xscore"] = predict_xscore(model, df[FEATURE_COLUMNS])
    df = normalize_drive_start_transition(df)
    print(f"  {len(df):,} regular season snaps")

    # === Compute GDS per team per season ===
    print("  Computing GDS decomposition...")
    st_baselines = compute_st_baselines(df)
    season_gds = compute_season_gds(df, st_baselines)
    print(f"  {len(season_gds)} team-seasons")

    # === Compute playoff outcomes ===
    print("  Computing playoff outcomes...")
    playoff_df = compute_playoff_outcomes(raw_df)
    print(f"  {playoff_df['made_playoffs'].sum()} playoff team-seasons")

    # === Merge ===
    merged = season_gds.merge(playoff_df, on=["season", "team"], how="left")
    merged["made_playoffs"] = merged["made_playoffs"].fillna(False)
    merged["playoff_wins"] = merged["playoff_wins"].fillna(0).astype(int)
    merged["won_super_bowl"] = merged["won_super_bowl"].fillna(False)
    merged["win_pct"] = merged["reg_wins"] / merged["reg_games"]
    merged["offense_share"] = merged.apply(compute_offense_share, axis=1)

    # === LAYER 1: Descriptive Analysis ===
    print("\n" + "=" * 70)
    print("LAYER 1: Descriptive Analysis")
    print("=" * 70)

    # Offense share quartiles vs playoff advancement
    merged["off_share_quartile"] = pd.qcut(
        merged["offense_share"], q=4, labels=["Q1 (Def)", "Q2", "Q3", "Q4 (Off)"]
    )
    print("\n--- Offense Share Quartile vs Playoff Advancement ---")
    quartile_stats = merged.groupby("off_share_quartile").agg(
        n=("team", "count"),
        avg_off_share=("offense_share", "mean"),
        playoff_rate=("made_playoffs", "mean"),
        avg_playoff_wins=("playoff_wins", "mean"),
        conf_rate=("playoff_wins", lambda x: (x >= 3).mean()),
        sb_rate=("won_super_bowl", "mean"),
    ).reset_index()
    print(f"{'Quartile':<12} {'N':<5} {'Avg Share':<10} {'Playoff%':<10} "
          f"{'Avg PW':<8} {'Conf+':<8} {'SB Win':<8}")
    print("-" * 65)
    for _, row in quartile_stats.iterrows():
        print(f"{row['off_share_quartile']:<12} {int(row['n']):<5} {row['avg_off_share']:<10.3f} "
              f"{row['playoff_rate']:<10.1%} {row['avg_playoff_wins']:<8.2f} "
              f"{row['conf_rate']:<8.1%} {row['sb_rate']:<8.1%}")

    # GDS strength quartiles
    merged["gds_quartile"] = pd.qcut(
        merged["gds_per_game"], q=4, labels=["Bottom", "Below Avg", "Above Avg", "Top"]
    )
    print("\n--- GDS Strength Quartile vs Playoff Advancement ---")
    gds_stats = merged.groupby("gds_quartile").agg(
        n=("team", "count"),
        playoff_rate=("made_playoffs", "mean"),
        avg_playoff_wins=("playoff_wins", "mean"),
        conf_rate=("playoff_wins", lambda x: (x >= 3).mean()),
        sb_rate=("won_super_bowl", "mean"),
    ).reset_index()
    print(f"{'Tier':<12} {'N':<5} {'Playoff%':<10} {'Avg PW':<8} {'Conf+':<8} {'SB Win':<8}")
    print("-" * 55)
    for _, row in gds_stats.iterrows():
        print(f"{row['gds_quartile']:<12} {int(row['n']):<5} {row['playoff_rate']:<10.1%} "
              f"{row['avg_playoff_wins']:<8.2f} {row['conf_rate']:<8.1%} {row['sb_rate']:<8.1%}")

    # Myth check: defense-dominant teams in playoffs
    def_dominant = merged[(merged["offense_share"] < -0.3) & (merged["made_playoffs"])]
    print(f"\n--- Myth Check: Defense-Dominant Teams (share < -0.3) in Playoffs ---")
    print(f"  Total defense-dominant playoff teams: {len(def_dominant)}")
    print(f"  Won 2+ playoff games: {(def_dominant['playoff_wins'] >= 2).sum()}/{len(def_dominant)}")
    print(f"  Won Super Bowl: {def_dominant['won_super_bowl'].sum()}/{len(def_dominant)}")

    # Super Bowl participants decomposition
    sb_teams = merged[merged["playoff_wins"] >= 3].sort_values(["season", "playoff_wins"], ascending=[True, False])
    print(f"\n--- Super Bowl Participants (3+ playoff wins) ---")
    print(f"{'Season':<8} {'Team':<6} {'Off/G':<8} {'Def/G':<8} {'Share':<8} {'PW':<4} {'SB':<4}")
    print("-" * 50)
    for _, row in sb_teams.iterrows():
        sb = "W" if row["won_super_bowl"] else "-"
        print(f"{int(row['season']):<8} {row['team']:<6} {row['off_xvoa_per_game']:.3f}  "
              f"{row['def_xvoa_per_game']:.3f}  {row['offense_share']:.3f}  "
              f"{int(row['playoff_wins']):<4} {sb}")

    # === LAYER 2: Statistical Confirmation ===
    print("\n" + "=" * 70)
    print("LAYER 2: Statistical Confirmation")
    print("=" * 70)

    # Spearman: offense_share vs playoff_wins
    playoff_teams_only = merged[merged["made_playoffs"]]
    rho, p = stats.spearmanr(playoff_teams_only["offense_share"], playoff_teams_only["playoff_wins"])
    print(f"\n--- Spearman: Offense Share vs Playoff Wins (playoff teams only) ---")
    print(f"  rho = {rho:.3f}, p = {p:.4f} (n={len(playoff_teams_only)})")
    sig = "SIGNIFICANT" if p < 0.05 else "not significant"
    print(f"  Result: {sig} at alpha=0.05")

    # Pearson: off_xvoa_per_game vs win%
    r, p2 = stats.pearsonr(merged["off_xvoa_per_game"], merged["win_pct"])
    print(f"\n--- Pearson: Off xVOA/game vs Win% (all teams) ---")
    print(f"  r = {r:.3f}, p = {p2:.6f} (n={len(merged)})")

    r_def, p_def = stats.pearsonr(merged["def_xvoa_per_game"], merged["win_pct"])
    print(f"\n--- Pearson: Def xVOA/game vs Win% (all teams) ---")
    print(f"  r = {r_def:.3f}, p = {p_def:.6f} (n={len(merged)})")

    print(f"\n  Offense correlation ({r:.3f}) vs Defense correlation ({r_def:.3f})")
    print(f"  Offense explains {r**2:.1%} of win% variance, Defense explains {r_def**2:.1%}")

    # Cohen's d: playoff wins for offense-dominant vs defense-dominant
    off_dom = playoff_teams_only[playoff_teams_only["offense_share"] > 0.3]["playoff_wins"]
    def_dom = playoff_teams_only[playoff_teams_only["offense_share"] < -0.3]["playoff_wins"]
    if len(off_dom) > 0 and len(def_dom) > 0:
        pooled_std = np.sqrt(((len(off_dom) - 1) * off_dom.std()**2 + (len(def_dom) - 1) * def_dom.std()**2)
                             / (len(off_dom) + len(def_dom) - 2))
        d = (off_dom.mean() - def_dom.mean()) / pooled_std if pooled_std > 0 else 0
        print(f"\n--- Cohen's d: Off-dominant vs Def-dominant playoff teams ---")
        print(f"  Off-dominant (share>0.3): n={len(off_dom)}, mean PW={off_dom.mean():.2f}")
        print(f"  Def-dominant (share<-0.3): n={len(def_dom)}, mean PW={def_dom.mean():.2f}")
        print(f"  Cohen's d = {d:.3f} ({'large' if abs(d) > 0.8 else 'medium' if abs(d) > 0.5 else 'small'})")

    # Logistic regression: P(won SB) ~ off_xvoa + def_xvoa
    print(f"\n--- Logistic Regression: P(Won Super Bowl) ---")
    X_lr = merged[["off_xvoa_per_game", "def_xvoa_per_game"]].copy()
    X_lr = sm.add_constant(X_lr)
    y_lr = merged["won_super_bowl"].astype(int)
    try:
        logit_model = sm.Logit(y_lr, X_lr).fit(disp=0)
        print(logit_model.summary2().tables[1].to_string())
        print(f"\n  Interpretation: For each +1.0 off_xvoa_per_game, odds of SB multiply by "
              f"{np.exp(logit_model.params['off_xvoa_per_game']):.2f}")
        print(f"  Interpretation: For each +1.0 def_xvoa_per_game, odds of SB multiply by "
              f"{np.exp(logit_model.params['def_xvoa_per_game']):.2f}")
    except Exception as e:
        print(f"  Logistic regression failed: {e}")

    # OLS regression: playoff_wins ~ offense_share + gds_per_game
    print(f"\n--- OLS Regression: Playoff Wins ~ Offense Share + GDS Strength ---")
    X_ols = merged[["offense_share", "gds_per_game"]].copy()
    X_ols = sm.add_constant(X_ols)
    y_ols = merged["playoff_wins"]
    ols_model = sm.OLS(y_ols, X_ols).fit()
    print(f"  R² = {ols_model.rsquared:.3f}")
    print(f"  offense_share coef = {ols_model.params['offense_share']:.3f} "
          f"(p={ols_model.pvalues['offense_share']:.4f})")
    print(f"  gds_per_game coef = {ols_model.params['gds_per_game']:.3f} "
          f"(p={ols_model.pvalues['gds_per_game']:.4f})")

    # === LAYER 3: Era Analysis ===
    print("\n" + "=" * 70)
    print("LAYER 3: Era Analysis (2018-2020 vs 2021-2024)")
    print("=" * 70)

    early = merged[merged["season"].between(2018, 2020)]
    late = merged[merged["season"].between(2021, 2024)]

    early_playoff = early[early["made_playoffs"]]
    late_playoff = late[late["made_playoffs"]]

    if len(early_playoff) > 5 and len(late_playoff) > 5:
        rho_early, p_early = stats.spearmanr(early_playoff["offense_share"], early_playoff["playoff_wins"])
        rho_late, p_late = stats.spearmanr(late_playoff["offense_share"], late_playoff["playoff_wins"])
        print(f"\n  2018-2020: Spearman rho = {rho_early:.3f} (p={p_early:.4f}, n={len(early_playoff)})")
        print(f"  2021-2024: Spearman rho = {rho_late:.3f} (p={p_late:.4f}, n={len(late_playoff)})")

        if rho_late > rho_early:
            print(f"\n  Finding: Offense share MORE predictive in recent era (+{rho_late-rho_early:.3f})")
            print(f"  Supports: Rule changes favoring offense have strengthened the effect")
        else:
            print(f"\n  Finding: No era strengthening detected (diff={rho_late-rho_early:.3f})")

    # Era win% correlations
    r_early_off, _ = stats.pearsonr(early["off_xvoa_per_game"], early["win_pct"])
    r_late_off, _ = stats.pearsonr(late["off_xvoa_per_game"], late["win_pct"])
    r_early_def, _ = stats.pearsonr(early["def_xvoa_per_game"], early["win_pct"])
    r_late_def, _ = stats.pearsonr(late["def_xvoa_per_game"], late["win_pct"])
    print(f"\n  Off xVOA vs Win%:  2018-2020 r={r_early_off:.3f}  |  2021-2024 r={r_late_off:.3f}")
    print(f"  Def xVOA vs Win%:  2018-2020 r={r_early_def:.3f}  |  2021-2024 r={r_late_def:.3f}")

    # === LAYER 4: "Both Sides" Finding ===
    print("\n" + "=" * 70)
    print("LAYER 4: The 'Both Sides' Finding — Offense Gets You There, Defense Wins It")
    print("=" * 70)

    top_gds = merged[merged["gds_per_game"] > merged["gds_per_game"].quantile(0.75)]
    off_median = top_gds["off_xvoa_per_game"].median()
    def_median = top_gds["def_xvoa_per_game"].median()

    top_gds = top_gds.copy()
    top_gds["quadrant"] = np.where(
        (top_gds["off_xvoa_per_game"] >= off_median) & (top_gds["def_xvoa_per_game"] >= def_median),
        "Elite Both",
        np.where(
            top_gds["off_xvoa_per_game"] >= off_median,
            "Offense Only",
            np.where(
                top_gds["def_xvoa_per_game"] >= def_median,
                "Defense Only",
                "Neither (below median both)"
            )
        )
    )

    print(f"\n  Among top-25% GDS teams (n={len(top_gds)}):")
    print(f"  Off median: {off_median:.3f}/game, Def median: {def_median:.3f}/game")
    print(f"\n{'Quadrant':<28} {'N':<5} {'Playoff%':<10} {'Avg PW':<8} {'Conf+':<8} {'SB Win':<8}")
    print("-" * 70)
    quad_stats = top_gds.groupby("quadrant").agg(
        n=("team", "count"),
        playoff_rate=("made_playoffs", "mean"),
        avg_pw=("playoff_wins", "mean"),
        conf_rate=("playoff_wins", lambda x: (x >= 3).mean()),
        sb_rate=("won_super_bowl", "mean"),
    ).reset_index().sort_values("avg_pw", ascending=False)
    for _, row in quad_stats.iterrows():
        print(f"  {row['quadrant']:<26} {int(row['n']):<5} {row['playoff_rate']:<10.1%} "
              f"{row['avg_pw']:<8.2f} {row['conf_rate']:<8.1%} {row['sb_rate']:<8.1%}")

    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("SUMMARY: Key Thesis Findings")
    print("=" * 70)
    print(f"\n  1. Offense share correlates with playoff wins: rho={rho:.3f} (p={p:.4f})")
    print(f"  2. Off xVOA explains {r**2:.1%} of win% variance, Defense explains {r_def**2:.1%}")
    print(f"  3. Defense-dominant playoff teams winning 2+ games: "
          f"{(def_dominant['playoff_wins'] >= 2).sum()}/{len(def_dominant)} "
          f"({(def_dominant['playoff_wins'] >= 2).mean():.0%})")
    print(f"  4. Verdict: 'Defense wins championships' is {'UNSUPPORTED' if rho > 0 else 'SUPPORTED'}")

    # === Export CSV ===
    import os
    os.makedirs("output", exist_ok=True)
    output_cols = ["season", "team", "games", "off_xvoa_per_game", "def_xvoa_per_game",
                   "gds_per_game", "offense_share", "reg_wins", "reg_games", "win_pct",
                   "made_playoffs", "playoff_wins", "won_super_bowl"]
    merged[output_cols].to_csv("output/archetype_data.csv", index=False)
    print(f"\n  Data exported to output/archetype_data.csv")

    print(f"\n{'=' * 70}")
    print("Analysis complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
