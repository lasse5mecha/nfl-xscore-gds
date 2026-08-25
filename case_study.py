"""Case Study: Play-by-play xScore narrative for a single game."""
import sys
import pandas as pd
import numpy as np
from src.data import (
    load_play_by_play, filter_offensive_snaps, add_drive_touchdown_target,
    engineer_features, FEATURE_COLUMNS, TARGET_COLUMN,
    normalize_drive_start_transition, compute_st_baselines, compute_st_value,
)
from src.model import load_model, predict_xscore, compute_xvoa, compute_game_deserved_score


def narrate_game(game_id: str, raw_df: pd.DataFrame, model, st_baselines: dict):
    """Full play-by-play narrative for a single game."""
    game_raw = raw_df[raw_df["game_id"] == game_id]
    if len(game_raw) == 0:
        print(f"Game {game_id} not found!")
        return

    meta = game_raw.iloc[0]
    home = meta["home_team"]
    away = meta["away_team"]
    home_score = int(meta["home_score"])
    away_score = int(meta["away_score"])
    week = int(meta["week"])
    winner = home if home_score > away_score else away

    # Process plays
    df = filter_offensive_snaps(game_raw)
    df = add_drive_touchdown_target(df)
    df = engineer_features(df)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    df["xscore"] = predict_xscore(model, df[FEATURE_COLUMNS])
    df = normalize_drive_start_transition(df)

    # Compute GDS
    game_xvoa = compute_xvoa(df[["game_id", "posteam", "drive", "xscore", "drive_td", "half_seconds_remaining"]])
    st_df = compute_st_value(df, st_baselines)
    gds_df = compute_game_deserved_score(game_xvoa, st_df)

    print("=" * 80)
    print(f"CASE STUDY: Week {week} — {away} @ {home}")
    print(f"Final Score: {home} {home_score} - {away} {away_score}")
    print("=" * 80)

    # GDS summary
    for _, row in gds_df.iterrows():
        team = row["posteam"]
        deserved = "DESERVED WIN" if row["gds"] > 0 and row["gds"] == gds_df["gds"].max() else ""
        print(f"  {team}: Off xVOA={row['offensive_xvoa']:+.2f}, "
              f"Def xVOA={row['defensive_xvoa']:+.2f}, "
              f"ST={row['st_value']:+.2f}, "
              f"GDS={row['gds']:+.2f} {deserved}")

    gds_winner = gds_df.loc[gds_df["gds"].idxmax(), "posteam"]
    if gds_winner != winner:
        print(f"\n  >>> UPSET: {gds_winner} deserved to win but {winner} won <<<")

    # Drive-by-drive narrative
    print(f"\n{'=' * 80}")
    print("DRIVE-BY-DRIVE NARRATIVE")
    print(f"{'=' * 80}")

    df_sorted = df.sort_values("half_seconds_remaining", ascending=False).copy()
    df_sorted["_xscore_next"] = df_sorted.groupby(["game_id", "posteam", "drive"])["xscore"].shift(-1)

    drives = df_sorted.groupby(["posteam", "drive"]).agg(
        n_plays=("xscore", "count"),
        start_xscore=("xscore", "first"),
        end_xscore=("xscore", "last"),
        max_xscore=("xscore", "max"),
        drive_td=("drive_td", "first"),
        start_yard=("yardline_100", "first"),
        time_start=("half_seconds_remaining", "max"),
        drive_start_type=("drive_start_type", "first"),
    ).reset_index()

    # Sort by time to get chronological order
    drives = drives.sort_values("time_start", ascending=False).reset_index(drop=True)

    cumulative_xvoa = {home: 0.0, away: 0.0}

    for i, (_, drive) in enumerate(drives.iterrows(), 1):
        team = drive["posteam"]
        td = "TD" if drive["drive_td"] == 1 else "No TD"
        start_yl = int(drive["start_yard"])

        # Delta for this drive
        if drive["drive_td"] == 1:
            drive_xvoa = (1.0 - drive["start_xscore"])
        else:
            drive_xvoa = (0.0 - drive["start_xscore"])

        # Approximate per-play contribution
        n = drive["n_plays"]
        cumulative_xvoa[team] += drive_xvoa

        arrow = "+" if drive_xvoa > 0 else ""
        momentum = ">>>" if drive_xvoa > 0.3 else ("<<<" if drive_xvoa < -0.3 else "---")

        time_min = drive["time_start"] // 60
        time_sec = drive["time_start"] % 60
        half = "1H" if drive["time_start"] > 0 else "2H"

        transition = drive["drive_start_type"] if pd.notna(drive["drive_start_type"]) else "?"

        print(f"  Drive {i:>2} | {team:<4} | {n:>2} plays | "
              f"Start: own {100-start_yl} ({transition}) | "
              f"xScore: {drive['start_xscore']:.2f}→{'1.00' if drive['drive_td']==1 else '0.00'} | "
              f"xVOA: {arrow}{drive_xvoa:.2f} {momentum} | {td}")

    # Final cumulative
    print(f"\n{'=' * 80}")
    print("CUMULATIVE xVOA (drive-level approximation)")
    print(f"{'=' * 80}")
    for team in [home, away]:
        print(f"  {team}: {cumulative_xvoa[team]:+.2f}")

    # Key momentum swings (biggest single-play deltas)
    print(f"\n{'=' * 80}")
    print("KEY MOMENTUM PLAYS (largest single-play xScore deltas)")
    print(f"{'=' * 80}")

    df_sorted["delta"] = df_sorted["_xscore_next"].fillna(0) - df_sorted["xscore"]
    # For last play of TD drives, delta goes to 1.0
    last_plays = df_sorted.groupby(["posteam", "drive"]).tail(1).index
    for idx in last_plays:
        if df_sorted.loc[idx, "drive_td"] == 1:
            df_sorted.loc[idx, "delta"] = 1.0 - df_sorted.loc[idx, "xscore"]
        else:
            df_sorted.loc[idx, "delta"] = 0.0 - df_sorted.loc[idx, "xscore"]

    top_plays = df_sorted.nlargest(5, "delta")
    worst_plays = df_sorted.nsmallest(5, "delta")

    print("\n  Top 5 Positive Plays (biggest xScore gains):")
    for _, play in top_plays.iterrows():
        desc = play.get("desc", "")[:80] if "desc" in play.index else ""
        print(f"    {play['posteam']} | xScore {play['xscore']:.2f}→{play['xscore']+play['delta']:.2f} "
              f"(+{play['delta']:.2f}) | {play.get('play_type', '')} | "
              f"{int(play['yardline_100'])} yd line")

    print("\n  Top 5 Negative Plays (biggest xScore drops):")
    for _, play in worst_plays.iterrows():
        print(f"    {play['posteam']} | xScore {play['xscore']:.2f}→{play['xscore']+play['delta']:.2f} "
              f"({play['delta']:+.2f}) | {play.get('play_type', '')} | "
              f"{int(play['yardline_100'])} yd line")


def main():
    model = load_model("models/xscore_v1.json")

    print("Loading 2024 data...")
    raw_df = load_play_by_play([2024])

    # Compute baselines from all 2024 reg season
    reg_df = raw_df[raw_df["season_type"] == "REG"]
    bl_df = filter_offensive_snaps(reg_df)
    bl_df = add_drive_touchdown_target(bl_df)
    bl_df = engineer_features(bl_df)
    bl_df = bl_df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    bl_df["xscore"] = predict_xscore(model, bl_df[FEATURE_COLUMNS])
    bl_df = normalize_drive_start_transition(bl_df)
    st_baselines = compute_st_baselines(bl_df)

    # Game 1: NYG vs WAS Week 2 (biggest upset of 2024)
    # nflfastR game_id format: YYYY_WW_AWAY_HOME
    narrate_game("2024_02_NYG_WAS", raw_df, model, st_baselines)

    print("\n\n")

    # Game 2: Super Bowl — KC vs PHI
    narrate_game("2024_22_KC_PHI", raw_df, model, st_baselines)


if __name__ == "__main__":
    main()
