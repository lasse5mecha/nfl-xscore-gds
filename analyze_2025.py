"""2025 NFL Season Analysis: xScore and xVOA rankings."""
import pandas as pd
import numpy as np
from src.data import (
    load_play_by_play, filter_offensive_snaps, add_drive_touchdown_target,
    engineer_features, FEATURE_COLUMNS, TARGET_COLUMN,
    normalize_drive_start_transition, compute_st_baselines, compute_st_value,
)
from src.model import load_model, predict_xscore, compute_game_xscore, compute_xvoa, compute_game_deserved_score


def main():
    print("=" * 70)
    print("2025 NFL Season — xScore Analysis")
    print("=" * 70)

    model = load_model("models/xscore_v1.json")

    print("\nLoading 2025 data...")
    raw_df = load_play_by_play([2025])
    df = filter_offensive_snaps(raw_df)
    df = add_drive_touchdown_target(df)
    df = engineer_features(df)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    df["xscore"] = predict_xscore(model, df[FEATURE_COLUMNS])
    print(f"  {len(df):,} offensive snaps analyzed")

    # === xVOA Rankings ===
    game_xvoa = compute_xvoa(df[["game_id", "posteam", "drive", "xscore", "drive_td", "half_seconds_remaining"]])

    # === GDS: Special Teams Baselines ===
    print("\nComputing ST baselines from training data (2018-2024)...")
    train_raw = load_play_by_play(list(range(2018, 2025)))
    train_filtered = filter_offensive_snaps(train_raw)
    train_filtered = add_drive_touchdown_target(train_filtered)
    train_filtered = engineer_features(train_filtered)
    train_filtered = train_filtered.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    train_filtered["xscore"] = predict_xscore(model, train_filtered[FEATURE_COLUMNS])
    train_filtered = normalize_drive_start_transition(train_filtered)
    st_baselines = compute_st_baselines(train_filtered)
    print(f"  ST Baselines: {', '.join(f'{k}={v:.3f}' for k, v in sorted(st_baselines.items()))}")

    # === GDS: Compute for 2025 ===
    df = normalize_drive_start_transition(df)
    st_value_df = compute_st_value(df, st_baselines)
    gds_df = compute_game_deserved_score(game_xvoa, st_value_df)

    season_xvoa = game_xvoa.groupby("posteam").agg(
        games=("game_id", "count"),
        total_xVOA=("xVOA", "sum"),
    ).reset_index()
    season_xvoa["xVOA_per_game"] = season_xvoa["total_xVOA"] / season_xvoa["games"]
    season_xvoa = season_xvoa.sort_values("total_xVOA", ascending=False)

    print("\n" + "=" * 70)
    print("Offensive Value Added (xVOA) Rankings")
    print("How much did each offense improve their TD probability beyond expectation?")
    print("=" * 70)
    print(f"{'Rank':<5} {'Team':<6} {'Games':<6} {'Total xVOA':<12} {'xVOA/Game':<10}")
    print("-" * 45)
    for i, (_, row) in enumerate(season_xvoa.iterrows(), 1):
        print(f"{i:<5} {row['posteam']:<6} {row['games']:<6} {row['total_xVOA']:<12.2f} {row['xVOA_per_game']:<10.3f}")

    # === GDS Rankings ===
    season_gds = gds_df.groupby("posteam").agg(
        games=("game_id", "count"),
        total_gds=("gds", "sum"),
        off_xvoa=("offensive_xvoa", "sum"),
        def_xvoa=("defensive_xvoa", "sum"),
        st_val=("st_value", "sum"),
    ).reset_index()
    season_gds["gds_per_game"] = season_gds["total_gds"] / season_gds["games"]
    season_gds = season_gds.sort_values("total_gds", ascending=False)

    print("\n" + "=" * 70)
    print("Game Deserved Score (GDS) Rankings — Three-Phase Decomposition")
    print("GDS = Offensive xVOA + Defensive xVOA + ST Value")
    print("=" * 70)
    print(f"{'Rank':<5} {'Team':<6} {'GDS':<9} {'Off xVOA':<10} {'Def xVOA':<10} {'ST Val':<8} {'GDS/G':<7}")
    print("-" * 60)
    for i, (_, row) in enumerate(season_gds.iterrows(), 1):
        print(f"{i:<5} {row['posteam']:<6} {row['total_gds']:<9.2f} "
              f"{row['off_xvoa']:<10.2f} {row['def_xvoa']:<10.2f} "
              f"{row['st_val']:<8.2f} {row['gds_per_game']:<7.3f}")

    # === Game-Winner Prediction Accuracy ===
    game_scores = raw_df.groupby("game_id").first()[
        ["home_team", "away_team", "home_score", "away_score"]
    ].reset_index()

    correct = 0
    total = 0
    upsets = []

    for game_id in game_xvoa["game_id"].unique():
        g = game_xvoa[game_xvoa["game_id"] == game_id]
        if len(g) != 2:
            continue
        t1, t2 = g.iloc[0], g.iloc[1]
        score_row = game_scores[game_scores["game_id"] == game_id]
        if len(score_row) == 0:
            continue
        score_row = score_row.iloc[0]

        pts = {}
        for t in [t1["posteam"], t2["posteam"]]:
            pts[t] = score_row["home_score"] if t == score_row["home_team"] else score_row["away_score"]

        xvoa_winner = t1["posteam"] if t1["xVOA"] > t2["xVOA"] else t2["posteam"]
        xvoa_loser = t2["posteam"] if t1["xVOA"] > t2["xVOA"] else t1["posteam"]
        pts_winner = (
            t1["posteam"] if pts[t1["posteam"]] > pts[t2["posteam"]]
            else (t2["posteam"] if pts[t2["posteam"]] > pts[t1["posteam"]] else "TIE")
        )

        if pts_winner == "TIE":
            continue
        total += 1
        if xvoa_winner == pts_winner:
            correct += 1
        else:
            xvoa_diff = abs(t1["xVOA"] - t2["xVOA"])
            upsets.append({
                "game_id": game_id,
                "xvoa_winner": xvoa_winner,
                "pts_winner": pts_winner,
                "xvoa_diff": xvoa_diff,
                "score": f"{int(pts[pts_winner])}-{int(pts[xvoa_winner])}",
            })

    print(f"\n{'=' * 70}")
    print(f"xVOA Game-Winner Prediction: {correct}/{total} = {correct/total:.1%}")
    print(f"{'=' * 70}")

    # === GDS Game-Winner Prediction ===
    gds_correct = 0
    gds_total = 0

    for game_id in gds_df["game_id"].unique():
        g = gds_df[gds_df["game_id"] == game_id]
        if len(g) != 2:
            continue
        t1, t2 = g.iloc[0], g.iloc[1]
        score_row = game_scores[game_scores["game_id"] == game_id]
        if len(score_row) == 0:
            continue
        score_row = score_row.iloc[0]

        pts = {}
        for t in [t1["posteam"], t2["posteam"]]:
            pts[t] = score_row["home_score"] if t == score_row["home_team"] else score_row["away_score"]

        gds_winner = t1["posteam"] if t1["gds"] > t2["gds"] else t2["posteam"]
        pts_winner = (
            t1["posteam"] if pts[t1["posteam"]] > pts[t2["posteam"]]
            else (t2["posteam"] if pts[t2["posteam"]] > pts[t1["posteam"]] else "TIE")
        )

        if pts_winner == "TIE":
            continue
        gds_total += 1
        if gds_winner == pts_winner:
            gds_correct += 1

    print(f"\n{'=' * 70}")
    print(f"GDS Game-Winner Prediction:  {gds_correct}/{gds_total} = {gds_correct/gds_total:.1%}")
    print(f"xVOA Game-Winner Prediction: {correct}/{total} = {correct/total:.1%}")
    print(f"Improvement: {(gds_correct/gds_total - correct/total)*100:+.1f} percentage points")
    print(f"{'=' * 70}")

    # === Biggest Upsets ===
    upsets_df = pd.DataFrame(upsets).sort_values("xvoa_diff", ascending=False)
    print(f"\n{'=' * 70}")
    print("Biggest Upsets (xVOA favorite lost)")
    print(f"{'=' * 70}")
    for _, row in upsets_df.head(10).iterrows():
        week = row["game_id"].split("_")[1]
        print(f"  Week {week}: {row['xvoa_winner']} deserved to win but {row['pts_winner']} won {row['score']}")

    # === Most Dominant Single-Game Performances ===
    print(f"\n{'=' * 70}")
    print("Most Dominant Single-Game Offensive Performances (xVOA)")
    print(f"{'=' * 70}")
    top_performances = game_xvoa.nlargest(10, "xVOA")
    for _, row in top_performances.iterrows():
        week = row["game_id"].split("_")[1]
        print(f"  {row['posteam']} Week {week}: xVOA = {row['xVOA']:.2f}")

    # === Most Competitive Games ===
    print(f"\n{'=' * 70}")
    print("Most Evenly Matched Games (smallest xVOA gap)")
    print(f"{'=' * 70}")
    game_pairs = []
    for game_id in game_xvoa["game_id"].unique():
        g = game_xvoa[game_xvoa["game_id"] == game_id]
        if len(g) != 2:
            continue
        t1, t2 = g.iloc[0], g.iloc[1]
        game_pairs.append({
            "game_id": game_id,
            "team1": t1["posteam"], "xvoa1": t1["xVOA"],
            "team2": t2["posteam"], "xvoa2": t2["xVOA"],
            "gap": abs(t1["xVOA"] - t2["xVOA"]),
        })
    pairs_df = pd.DataFrame(game_pairs).sort_values("gap")
    for _, row in pairs_df.head(10).iterrows():
        week = row["game_id"].split("_")[1]
        print(f"  Week {week}: {row['team1']} ({row['xvoa1']:.2f}) vs {row['team2']} ({row['xvoa2']:.2f}) — gap: {row['gap']:.3f}")

    # === Luck Rankings: GDS-predicted wins vs actual wins ===
    team_gds_wins = {}
    team_actual_wins = {}

    for game_id in gds_df["game_id"].unique():
        g = gds_df[gds_df["game_id"] == game_id]
        if len(g) != 2:
            continue
        t1, t2 = g.iloc[0], g.iloc[1]
        score_row = game_scores[game_scores["game_id"] == game_id]
        if len(score_row) == 0:
            continue
        score_row = score_row.iloc[0]

        pts = {}
        for t in [t1["posteam"], t2["posteam"]]:
            pts[t] = score_row["home_score"] if t == score_row["home_team"] else score_row["away_score"]

        gds_winner = t1["posteam"] if t1["gds"] > t2["gds"] else t2["posteam"]
        gds_loser = t2["posteam"] if t1["gds"] > t2["gds"] else t1["posteam"]
        team_gds_wins[gds_winner] = team_gds_wins.get(gds_winner, 0) + 1
        team_gds_wins.setdefault(gds_loser, 0)

        if pts[t1["posteam"]] > pts[t2["posteam"]]:
            actual_winner = t1["posteam"]
            actual_loser = t2["posteam"]
        elif pts[t2["posteam"]] > pts[t1["posteam"]]:
            actual_winner = t2["posteam"]
            actual_loser = t1["posteam"]
        else:
            continue
        team_actual_wins[actual_winner] = team_actual_wins.get(actual_winner, 0) + 1
        team_actual_wins.setdefault(actual_loser, 0)

    luck_df = pd.DataFrame({
        "team": list(set(team_gds_wins.keys()) | set(team_actual_wins.keys())),
    })
    luck_df["gds_wins"] = luck_df["team"].map(team_gds_wins).fillna(0)
    luck_df["actual_wins"] = luck_df["team"].map(team_actual_wins).fillna(0)
    luck_df["luck"] = luck_df["actual_wins"] - luck_df["gds_wins"]
    luck_df = luck_df.sort_values("luck", ascending=False)

    print(f"\n{'=' * 70}")
    print("Luck Rankings (Actual Wins - GDS-Deserved Wins)")
    print("Positive = lucky (won more than deserved), Negative = unlucky")
    print(f"{'=' * 70}")
    print(f"{'Rank':<5} {'Team':<6} {'Actual W':<10} {'GDS W':<8} {'Luck':<6}")
    print("-" * 40)
    for i, (_, row) in enumerate(luck_df.iterrows(), 1):
        sign = "+" if row["luck"] >= 0 else ""
        print(f"{i:<5} {row['team']:<6} {int(row['actual_wins']):<10} "
              f"{int(row['gds_wins']):<8} {sign}{int(row['luck']):<5}")

    print(f"\n{'=' * 70}")
    print("Analysis complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
