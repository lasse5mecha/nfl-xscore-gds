"""Exploratory: GDS team archetypes vs playoff success (2018-2024)."""
import pandas as pd
import numpy as np
from src.data import (
    load_play_by_play, filter_offensive_snaps, add_drive_touchdown_target,
    engineer_features, FEATURE_COLUMNS, TARGET_COLUMN,
    normalize_drive_start_transition, compute_st_baselines, compute_st_value,
)
from src.model import load_model, predict_xscore, compute_xvoa, compute_game_deserved_score


def classify_archetype(row):
    """Classify team into archetype based on GDS component dominance."""
    off = row["off_xvoa_per_game"]
    deff = row["def_xvoa_per_game"]
    total = abs(off) + abs(deff)
    if total < 0.1:
        return "Balanced (weak)"
    off_pct = off / total if total > 0 else 0
    if off_pct > 0.6:
        return "Offense-dominant"
    elif off_pct < -0.6:
        return "Defense-dominant (negative)"
    elif deff / total > 0.6:
        return "Defense-dominant"
    elif abs(off_pct - 0.5) < 0.15 and row["gds_per_game"] > 0.3:
        return "Balanced (strong)"
    elif row["gds_per_game"] > 0.3:
        if off > deff:
            return "Offense-dominant"
        else:
            return "Defense-dominant"
    else:
        return "Balanced (weak)"


def main():
    print("=" * 70)
    print("GDS Archetype Analysis: Team Builds vs Playoff Success (2018-2024)")
    print("=" * 70)

    model = load_model("models/xscore_v1.json")

    print("\nLoading all data (2018-2024)...")
    raw_df = load_play_by_play(list(range(2018, 2025)))

    # Split regular season (for GDS computation) and playoffs (for outcomes)
    reg_df = raw_df[raw_df["season_type"] == "REG"].copy()
    post_df = raw_df[raw_df["season_type"] == "POST"].copy()

    print("Processing regular season plays...")
    df = filter_offensive_snaps(reg_df)
    df = add_drive_touchdown_target(df)
    df = engineer_features(df)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    df["xscore"] = predict_xscore(model, df[FEATURE_COLUMNS])
    df = normalize_drive_start_transition(df)
    print(f"  {len(df):,} regular season snaps")

    # Compute ST baselines from full training data
    st_baselines = compute_st_baselines(df)
    print(f"  ST Baselines computed")

    # Compute GDS per team per season
    all_season_gds = []

    for season in range(2018, 2025):
        season_df = df[df["season"] == season]
        if len(season_df) == 0:
            continue

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
        all_season_gds.append(season_summary)

    gds_all = pd.concat(all_season_gds, ignore_index=True)
    print(f"  {len(gds_all)} team-seasons computed")

    # Determine playoff outcomes
    # Weeks: 19=Wild Card, 20=Divisional, 21=Conference, 22=Super Bowl
    playoff_teams = []
    for season in range(2018, 2025):
        season_post = post_df[post_df["season"] == season]
        if len(season_post) == 0:
            continue

        teams_in_playoffs = set(season_post["home_team"].unique()) | set(season_post["away_team"].unique())

        for team in teams_in_playoffs:
            team_games = season_post[
                (season_post["home_team"] == team) | (season_post["away_team"] == team)
            ]
            max_week = team_games["week"].max()

            # Did they win their last game?
            last_game = team_games[team_games["week"] == max_week].iloc[0]
            if team == last_game["home_team"]:
                won_last = last_game["home_score"] > last_game["away_score"]
            else:
                won_last = last_game["away_score"] > last_game["home_score"]

            # Determine furthest round
            if max_week == 22 and won_last:
                furthest = "Super Bowl Winner"
            elif max_week == 22:
                furthest = "Super Bowl Loss"
            elif max_week == 21 and won_last:
                furthest = "Super Bowl Loss"  # Won conf = went to SB
            elif max_week == 21:
                furthest = "Conference"
            elif max_week == 20 and won_last:
                furthest = "Conference"  # Won div = went to conf
            elif max_week == 20:
                furthest = "Divisional"
            elif max_week == 19 and won_last:
                furthest = "Divisional"  # Won WC = went to div
            else:
                furthest = "Wild Card"

            # Fix: won_last in week 21 means they went to SB (week 22)
            # Let's simplify: track the max week they PLAYED in
            # week 19 = Wild Card round, week 20 = Divisional, week 21 = Conference, week 22 = Super Bowl
            playoff_teams.append({
                "season": season,
                "team": team,
                "max_week": int(max_week),
                "won_championship": max_week == 22 and won_last,
            })

    playoff_df = pd.DataFrame(playoff_teams)
    print(f"  {len(playoff_df)} playoff team-seasons identified")

    # Merge GDS with playoff outcomes
    gds_all = gds_all.rename(columns={"posteam": "team"})
    merged = gds_all.merge(playoff_df, on=["season", "team"], how="left")
    merged["made_playoffs"] = merged["max_week"].notna()
    merged["won_division_round"] = merged["max_week"] >= 20
    merged["made_conference"] = merged["max_week"] >= 21
    merged["made_super_bowl"] = merged["max_week"] >= 22
    merged["won_super_bowl"] = merged["won_championship"].fillna(False)

    # Classify archetypes
    merged["archetype"] = merged.apply(classify_archetype, axis=1)

    # Simpler classification: just use off/def ratio
    merged["off_share"] = merged["off_xvoa"] / (merged["off_xvoa"].abs() + merged["def_xvoa"].abs() + 0.01)

    # Tier teams by GDS
    merged["gds_tier"] = pd.qcut(merged["gds_per_game"], q=4, labels=["Bottom", "Below Avg", "Above Avg", "Top"])

    # === Results ===
    print("\n" + "=" * 70)
    print("1. GDS Tier vs Playoff Rate")
    print("=" * 70)
    tier_stats = merged.groupby("gds_tier").agg(
        n=("team", "count"),
        playoff_rate=("made_playoffs", "mean"),
        div_round_rate=("won_division_round", "mean"),
        conf_rate=("made_conference", "mean"),
        sb_rate=("made_super_bowl", "mean"),
        sb_win_rate=("won_super_bowl", "mean"),
    ).reset_index()
    print(f"{'Tier':<12} {'N':<5} {'Playoffs':<10} {'Div+':<8} {'Conf+':<8} {'SB':<6} {'Champ':<6}")
    print("-" * 55)
    for _, row in tier_stats.iterrows():
        print(f"{row['gds_tier']:<12} {int(row['n']):<5} {row['playoff_rate']:.1%}     "
              f"{row['div_round_rate']:.1%}   {row['conf_rate']:.1%}   "
              f"{row['sb_rate']:.1%} {row['sb_win_rate']:.1%}")

    print("\n" + "=" * 70)
    print("2. Archetype vs Playoff Success (playoff teams only)")
    print("=" * 70)
    playoff_only = merged[merged["made_playoffs"]].copy()
    arch_stats = playoff_only.groupby("archetype").agg(
        n=("team", "count"),
        avg_gds=("gds_per_game", "mean"),
        div_round_rate=("won_division_round", "mean"),
        conf_rate=("made_conference", "mean"),
        sb_rate=("made_super_bowl", "mean"),
        sb_win_rate=("won_super_bowl", "mean"),
    ).reset_index().sort_values("conf_rate", ascending=False)
    print(f"{'Archetype':<25} {'N':<5} {'Avg GDS/G':<10} {'Div+':<8} {'Conf+':<8} {'SB':<6} {'Champ':<6}")
    print("-" * 75)
    for _, row in arch_stats.iterrows():
        print(f"{row['archetype']:<25} {int(row['n']):<5} {row['avg_gds']:.3f}    "
              f"{row['div_round_rate']:.1%}   {row['conf_rate']:.1%}   "
              f"{row['sb_rate']:.1%} {row['sb_win_rate']:.1%}")

    print("\n" + "=" * 70)
    print("3. Offense-Dominant vs Defense-Dominant (Top-16 GDS teams only)")
    print("=" * 70)
    top_half = merged[merged["gds_per_game"] > merged["gds_per_game"].median()].copy()
    top_half["build"] = np.where(
        top_half["off_xvoa_per_game"] > top_half["def_xvoa_per_game"],
        "Offense > Defense",
        "Defense > Offense",
    )
    build_stats = top_half.groupby("build").agg(
        n=("team", "count"),
        playoff_rate=("made_playoffs", "mean"),
        conf_rate=("made_conference", "mean"),
        sb_rate=("made_super_bowl", "mean"),
        sb_win_rate=("won_super_bowl", "mean"),
    ).reset_index()
    print(f"{'Build':<20} {'N':<5} {'Playoffs':<10} {'Conf+':<8} {'SB':<6} {'Champ':<6}")
    print("-" * 55)
    for _, row in build_stats.iterrows():
        print(f"{row['build']:<20} {int(row['n']):<5} {row['playoff_rate']:.1%}     "
              f"{row['conf_rate']:.1%}   {row['sb_rate']:.1%} {row['sb_win_rate']:.1%}")

    print("\n" + "=" * 70)
    print("4. Super Bowl Participants — GDS Decomposition")
    print("=" * 70)
    sb_teams = merged[merged["made_super_bowl"]].sort_values("season")
    print(f"{'Season':<8} {'Team':<6} {'GDS/G':<8} {'Off/G':<8} {'Def/G':<8} {'Won':<5}")
    print("-" * 50)
    for _, row in sb_teams.iterrows():
        won = "Y" if row["won_super_bowl"] else "N"
        print(f"{int(row['season']):<8} {row['team']:<6} {row['gds_per_game']:.3f}  "
              f"{row['off_xvoa_per_game']:.3f}  {row['def_xvoa_per_game']:.3f}  {won}")

    print("\n" + "=" * 70)
    print("5. Conference Championship+ Teams — What GDS Profile Gets You There?")
    print("=" * 70)
    conf_teams = merged[merged["made_conference"]].sort_values("gds_per_game", ascending=False)
    avg_off = conf_teams["off_xvoa_per_game"].mean()
    avg_def = conf_teams["def_xvoa_per_game"].mean()
    print(f"  N = {len(conf_teams)} team-seasons reached Conference Championship+")
    print(f"  Avg Offensive xVOA/game: {avg_off:.3f}")
    print(f"  Avg Defensive xVOA/game: {avg_def:.3f}")
    print(f"  Offense-led (Off > Def): {(conf_teams['off_xvoa_per_game'] > conf_teams['def_xvoa_per_game']).sum()}/{len(conf_teams)}")
    print(f"  Defense-led (Def > Off): {(conf_teams['def_xvoa_per_game'] > conf_teams['off_xvoa_per_game']).sum()}/{len(conf_teams)}")

    print(f"\n{'=' * 70}")
    print("Exploration complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
