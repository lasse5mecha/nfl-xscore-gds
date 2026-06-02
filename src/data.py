import nfl_data_py as nfl
import numpy as np
import pandas as pd

OUTCOME_CLASSES = ["punt_other", "td", "fg", "turnover"]
OUTCOME_TO_INDEX = {label: i for i, label in enumerate(OUTCOME_CLASSES)}


def load_play_by_play(seasons: list[int]) -> pd.DataFrame:
    """Load nflfastR play-by-play data for given seasons."""
    df = nfl.import_pbp_data(seasons)
    return df


def add_drive_touchdown_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary target: did this drive end in an offensive TD by the possession team?"""
    df = df.copy()
    df["__off_td"] = ((df["touchdown"] == 1) & (df["td_team"] == df["posteam"])).astype(int)
    drive_has_td = (
        df.groupby(["game_id", "drive"])["__off_td"]
        .transform("max")
    )
    df["drive_td"] = drive_has_td
    df = df.drop(columns=["__off_td"])
    return df


def add_drive_outcome_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add 4-class drive outcome: td, fg, turnover, punt_other.

    Uses fixed_drive_result (nflfastR) when available — this column is present
    on every play in the drive and correctly identifies the drive outcome even
    after filtering to offensive snaps only. Falls back to play-level flags
    if fixed_drive_result is not available (e.g., in unit tests).
    """
    df = df.copy()

    if "fixed_drive_result" in df.columns:
        _RESULT_MAP = {
            "Touchdown": "td",
            "Field goal": "fg",
            "Turnover": "turnover",
            "Turnover on downs": "turnover",
            "Punt": "punt_other",
            "End of half": "punt_other",
            "Missed field goal": "punt_other",
            "Opp touchdown": "turnover",
            "Safety": "punt_other",
        }
        drive_info = df.groupby(["game_id", "drive"])["fixed_drive_result"].first().reset_index()
        drive_info["drive_outcome_label"] = drive_info["fixed_drive_result"].map(_RESULT_MAP).fillna("punt_other")
        drive_info["drive_outcome"] = drive_info["drive_outcome_label"].map(OUTCOME_TO_INDEX)
        df = df.merge(
            drive_info[["game_id", "drive", "drive_outcome_label", "drive_outcome"]],
            on=["game_id", "drive"],
            how="left",
        )
    else:
        df["__off_td_flag"] = (
            (df["touchdown"] == 1) & (df["td_team"] == df["posteam"])
        ).astype(int)
        df["__fg_flag"] = (df["field_goal_result"] == "made").astype(int)
        df["__turnover_flag"] = (
            (df["interception"] == 1)
            | (df["fumble_lost"] == 1)
            | (df["fourth_down_failed"] == 1)
        ).astype(int)

        drive_info = df.groupby(["game_id", "drive"]).agg(
            has_off_td=("__off_td_flag", "max"),
            has_fg=("__fg_flag", "max"),
            has_turnover=("__turnover_flag", "max"),
        ).reset_index()

        def _classify(row):
            if row["has_off_td"] == 1:
                return "td"
            elif row["has_fg"] == 1:
                return "fg"
            elif row["has_turnover"] == 1:
                return "turnover"
            return "punt_other"

        drive_info["drive_outcome_label"] = drive_info.apply(_classify, axis=1)
        drive_info["drive_outcome"] = drive_info["drive_outcome_label"].map(OUTCOME_TO_INDEX)

        df = df.merge(
            drive_info[["game_id", "drive", "drive_outcome_label", "drive_outcome"]],
            on=["game_id", "drive"],
            how="left",
        )
        df = df.drop(columns=["__off_td_flag", "__fg_flag", "__turnover_flag"])
    return df


def filter_offensive_snaps(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only legitimate offensive snaps (pass, run, sack)."""
    mask = (
        df["play_type"].isin(["pass", "run"])
        & (df["two_point_attempt"] == 0)
        & (df["qb_kneel"] == 0)
        & (df["qb_spike"] == 0)
        & (df["aborted_play"] == 0)
        & df["down"].notna()
    )
    return df[mask].reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model features from raw play-by-play columns."""
    df = df.copy()
    df["score_diff"] = df["posteam_score"] - df["defteam_score"]
    df["goal_to_go"] = (df["yardline_100"] <= df["ydstogo"]).astype(int)
    df["red_zone"] = (df["yardline_100"] <= 20).astype(int)
    return df


def engineer_phase2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add team quality features: offense EPA, defense red zone TD%, home/away."""
    df = df.copy()
    season_epa = df.groupby(["season", "posteam"])["epa"].mean().reset_index()
    season_epa.columns = ["season", "posteam", "offense_epa"]
    df = df.merge(season_epa, on=["season", "posteam"], how="left")

    rz_plays = df[df["yardline_100"] <= 20].copy()
    rz_plays["off_td"] = ((rz_plays["touchdown"] == 1) & (rz_plays["td_team"] == rz_plays["posteam"])).astype(int)
    rz_td_rate = rz_plays.groupby(["season", "defteam"])["off_td"].mean().reset_index()
    rz_td_rate.columns = ["season", "defteam", "defense_rz_td_pct"]
    df = df.merge(rz_td_rate, on=["season", "defteam"], how="left")

    df["is_home"] = (df["posteam_type"] == "home").astype(int)
    df["defense_rz_td_pct"] = df["defense_rz_td_pct"].fillna(df["defense_rz_td_pct"].mean())
    df["offense_epa"] = df["offense_epa"].fillna(0)
    return df


def compute_rolling_epa(df: pd.DataFrame, decay: float = 0.5, prior_k: int = 4) -> pd.DataFrame:
    """Compute rolling exponentially-weighted EPA/play per team, with Bayesian prior."""
    df = df.copy()

    game_epa_off = (
        df.groupby(["season", "week", "game_id", "posteam"])["epa"]
        .mean()
        .reset_index()
        .rename(columns={"epa": "game_epa_off"})
    )

    game_epa_def = (
        df.groupby(["season", "week", "game_id", "defteam"])["epa"]
        .mean()
        .reset_index()
        .rename(columns={"epa": "game_epa_def"})
    )

    def _weighted_avg(game_epas: list[float], decay: float, prior_k: int) -> float:
        if not game_epas:
            return 0.0
        weights = [decay ** i for i in range(len(game_epas) - 1, -1, -1)]
        weighted_sum = sum(w * e for w, e in zip(weights, game_epas))
        total_weight = sum(weights)
        raw_epa = weighted_sum / total_weight
        n_games = len(game_epas)
        adjusted = (n_games * raw_epa + prior_k * 0.0) / (n_games + prior_k)
        return adjusted

    offense_lookup = {}
    for (season, team), group in game_epa_off.groupby(["season", "posteam"]):
        sorted_games = group.sort_values("week")
        prior_epas: list[float] = []
        for _, row in sorted_games.iterrows():
            offense_lookup[(season, team, row["week"])] = _weighted_avg(prior_epas, decay, prior_k)
            prior_epas.append(row["game_epa_off"])

    defense_lookup = {}
    for (season, team), group in game_epa_def.groupby(["season", "defteam"]):
        sorted_games = group.sort_values("week")
        prior_epas: list[float] = []
        for _, row in sorted_games.iterrows():
            defense_lookup[(season, team, row["week"])] = _weighted_avg(prior_epas, decay, prior_k)
            prior_epas.append(row["game_epa_def"])

    df["rolling_offense_epa"] = df.apply(
        lambda r: offense_lookup.get((r["season"], r["posteam"], r["week"]), 0.0), axis=1
    )
    df["rolling_defense_epa"] = df.apply(
        lambda r: defense_lookup.get((r["season"], r["defteam"], r["week"]), 0.0), axis=1
    )

    return df


def compute_momentum_epa(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Compute in-game momentum: mean EPA over last N offensive plays by posteam."""
    df = df.copy()
    momentum_values = []

    for _, group in df.groupby(["game_id", "posteam"]):
        epas = group["epa"].tolist()
        for i in range(len(epas)):
            if i == 0:
                momentum_values.append((group.index[i], 0.0))
            else:
                lookback = epas[max(0, i - window):i]
                momentum_values.append((group.index[i], np.mean(lookback)))

    momentum_df = pd.DataFrame(momentum_values, columns=["idx", "momentum_epa"])
    momentum_df = momentum_df.set_index("idx")
    df["momentum_epa"] = momentum_df["momentum_epa"]
    return df


def engineer_phase2_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Phase 2 features: rolling team EPA, momentum, and home/away."""
    df = compute_rolling_epa(df)
    df = compute_momentum_epa(df)
    df["is_home"] = (df["posteam_type"] == "home").astype(int)
    return df


DRIVE_START_TRANSITION_MAP = {
    "MUFFED_PUNT": "PUNT",
    "BLOCKED_PUNT": "PUNT",
    "BLOCKED_PUNT,_DOWNS": "PUNT",
    "BLOCKED_FG": "MISSED_FG",
    "BLOCKED_FG,_DOWNS": "MISSED_FG",
    "MUFFED_FG": "MISSED_FG",
    "ONSIDE_KICK": "KICKOFF",
    "MUFFED_KICKOFF": "KICKOFF",
    "OWN_KICKOFF": "KICKOFF",
}


def normalize_drive_start_transition(df: pd.DataFrame) -> pd.DataFrame:
    """Map rare drive start transition types to their parent category."""
    df = df.copy()
    df["drive_start_type"] = df["drive_start_transition"].map(
        lambda x: DRIVE_START_TRANSITION_MAP.get(x, x) if pd.notna(x) else x
    )
    return df


def compute_st_baselines(df: pd.DataFrame, model=None) -> dict[str, float]:
    """Compute expected starting xScore per drive transition type.

    Uses the mean actual xScore from first plays of each drive in the training
    data, grouped by transition type. This captures real game context (time,
    score) rather than a synthetic neutral reference.
    """
    first_plays = df[df["_is_first_play"]].copy() if "_is_first_play" in df.columns else (
        df.groupby(["game_id", "posteam", "drive"]).first().reset_index()
    )

    baselines = (
        first_plays.groupby("drive_start_type")["xscore"]
        .mean()
        .to_dict()
    )

    return {k: v for k, v in baselines.items() if pd.notna(k)}


def compute_st_value(df: pd.DataFrame, baselines: dict[str, float]) -> pd.DataFrame:
    """Compute ST Value per team per game: actual start xScore minus expected."""
    df = df.sort_values(
        ["game_id", "posteam", "drive", "half_seconds_remaining"],
        ascending=[True, True, True, False],
    ).copy()

    first_plays = df.groupby(["game_id", "posteam", "drive"]).first().reset_index()

    first_plays["_expected_xscore"] = first_plays["drive_start_type"].map(baselines)
    first_plays["_st_delta"] = first_plays["xscore"] - first_plays["_expected_xscore"]
    first_plays["_st_delta"] = first_plays["_st_delta"].fillna(0.0)

    st_value = (
        first_plays.groupby(["game_id", "posteam"])["_st_delta"]
        .sum()
        .reset_index()
        .rename(columns={"_st_delta": "st_value"})
    )
    return st_value


def compute_playoff_outcomes(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Extract playoff outcomes per team per season from raw play-by-play data.

    Returns DataFrame with columns: season, team, made_playoffs, playoff_wins,
    won_super_bowl, reg_wins, reg_games.
    """
    seasons = raw_df["season"].unique()
    results = []

    for season in seasons:
        season_df = raw_df[raw_df["season"] == season]

        # Regular season wins per team
        reg_games = season_df[season_df["season_type"] == "REG"]
        reg_scores = reg_games.groupby("game_id").first()[
            ["home_team", "away_team", "home_score", "away_score"]
        ].reset_index()

        team_reg_wins = {}
        team_reg_games = {}
        for _, game in reg_scores.iterrows():
            for team in [game["home_team"], game["away_team"]]:
                team_reg_games[team] = team_reg_games.get(team, 0) + 1
                if team == game["home_team"] and game["home_score"] > game["away_score"]:
                    team_reg_wins[team] = team_reg_wins.get(team, 0) + 1
                elif team == game["away_team"] and game["away_score"] > game["home_score"]:
                    team_reg_wins[team] = team_reg_wins.get(team, 0) + 1
                else:
                    team_reg_wins.setdefault(team, 0)

        # Playoff outcomes
        post_games = season_df[season_df["season_type"] == "POST"]
        post_scores = post_games.groupby("game_id").first()[
            ["week", "home_team", "away_team", "home_score", "away_score"]
        ].reset_index()

        playoff_teams = set()
        team_playoff_wins = {}
        sb_winner = None
        sb_week = post_scores["week"].max() if len(post_scores) > 0 else None

        for _, game in post_scores.iterrows():
            home = game["home_team"]
            away = game["away_team"]
            playoff_teams.add(home)
            playoff_teams.add(away)
            team_playoff_wins.setdefault(home, 0)
            team_playoff_wins.setdefault(away, 0)

            if game["home_score"] > game["away_score"]:
                team_playoff_wins[home] += 1
                if game["week"] == sb_week:
                    sb_winner = home
            elif game["away_score"] > game["home_score"]:
                team_playoff_wins[away] += 1
                if game["week"] == sb_week:
                    sb_winner = away

        # Build results for all teams in this season (union of REG + POST teams)
        all_teams = set(team_reg_wins.keys()) | playoff_teams
        for team in all_teams:
            results.append({
                "season": season,
                "team": team,
                "made_playoffs": team in playoff_teams,
                "playoff_wins": team_playoff_wins.get(team, 0),
                "won_super_bowl": team == sb_winner,
                "reg_wins": team_reg_wins.get(team, 0),
                "reg_games": team_reg_games.get(team, 0),
            })

    return pd.DataFrame(results)


PHASE2_FEATURE_COLUMNS = [
    "down", "ydstogo", "yardline_100", "score_diff",
    "half_seconds_remaining", "goal_to_go", "red_zone",
    "rolling_offense_epa", "rolling_defense_epa",
    "momentum_epa", "is_home",
]


def split_by_season(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split data: 2018-2024 train, 2025 test, 2014-2017 OOD validation."""
    train = df[df["season"].between(2018, 2024)].reset_index(drop=True)
    test = df[df["season"] == 2025].reset_index(drop=True)
    ood = df[df["season"].between(2014, 2017)].reset_index(drop=True)
    return {"train": train, "test": test, "ood": ood}


def prepare_dataset(seasons: list[int] | None = None) -> dict[str, pd.DataFrame]:
    """Full pipeline: load -> filter -> target -> features -> split."""
    if seasons is None:
        seasons = list(range(2014, 2026))
    df = load_play_by_play(seasons)
    df = filter_offensive_snaps(df)
    df = add_drive_touchdown_target(df)
    df = engineer_features(df)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    return split_by_season(df)


def prepare_phase2_dataset(seasons: list[int] | None = None) -> dict[str, pd.DataFrame]:
    """Full Phase 2 pipeline: load -> filter -> target -> features -> rolling EPA -> split."""
    if seasons is None:
        seasons = list(range(2014, 2026))
    df = load_play_by_play(seasons)
    df = filter_offensive_snaps(df)
    df = add_drive_touchdown_target(df)
    df = engineer_features(df)
    df = engineer_phase2_rolling_features(df)
    df = df.dropna(subset=PHASE2_FEATURE_COLUMNS + [TARGET_COLUMN])
    return split_by_season(df)


FEATURE_COLUMNS = [
    "down", "ydstogo", "yardline_100", "score_diff",
    "half_seconds_remaining", "goal_to_go", "red_zone",
]

TARGET_COLUMN = "drive_td"
