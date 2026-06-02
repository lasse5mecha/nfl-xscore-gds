import pandas as pd
from src.data import load_play_by_play
from src.data import filter_offensive_snaps
from src.data import add_drive_touchdown_target
from src.data import engineer_features
from src.data import split_by_season
from src.data import prepare_dataset, FEATURE_COLUMNS, TARGET_COLUMN
from src.data import engineer_phase2_features


def test_load_play_by_play_returns_dataframe():
    df = load_play_by_play(seasons=[2024])
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 10000
    assert "play_type" in df.columns
    assert "yardline_100" in df.columns
    assert "down" in df.columns


def test_filter_offensive_snaps_removes_kickoffs():
    df = pd.DataFrame({
        "play_type": ["pass", "run", "kickoff", "punt", "extra_point", "pass"],
        "two_point_attempt": [0, 0, 0, 0, 0, 0],
        "aborted_play": [0, 0, 0, 0, 0, 0],
        "qb_kneel": [0, 0, 0, 0, 0, 0],
        "qb_spike": [0, 0, 0, 0, 0, 0],
        "down": [1, 2, None, None, None, 3],
    })
    result = filter_offensive_snaps(df)
    assert len(result) == 3
    assert set(result["play_type"].unique()) == {"pass", "run"}


def test_filter_removes_two_point_attempts():
    df = pd.DataFrame({
        "play_type": ["pass", "pass"],
        "two_point_attempt": [0, 1],
        "aborted_play": [0, 0],
        "qb_kneel": [0, 0],
        "qb_spike": [0, 0],
        "down": [1, 1],
    })
    result = filter_offensive_snaps(df)
    assert len(result) == 1


def test_filter_removes_kneels_and_spikes():
    df = pd.DataFrame({
        "play_type": ["pass", "run", "run"],
        "two_point_attempt": [0, 0, 0],
        "aborted_play": [0, 0, 0],
        "qb_kneel": [0, 1, 0],
        "qb_spike": [0, 0, 1],
        "down": [1, 2, 3],
    })
    result = filter_offensive_snaps(df)
    assert len(result) == 1


def test_drive_touchdown_target_marks_td_drives():
    df = pd.DataFrame({
        "game_id": ["G1", "G1", "G1", "G1", "G1"],
        "drive": [1, 1, 1, 2, 2],
        "touchdown": [0, 0, 1, 0, 0],
        "td_team": ["KC", "KC", "KC", "KC", "KC"],
        "posteam": ["KC", "KC", "KC", "KC", "KC"],
    })
    result = add_drive_touchdown_target(df)
    assert result["drive_td"].tolist() == [1, 1, 1, 0, 0]


def test_drive_touchdown_excludes_defensive_tds():
    df = pd.DataFrame({
        "game_id": ["G1", "G1"],
        "drive": [1, 1],
        "touchdown": [0, 1],
        "td_team": [None, "BUF"],
        "posteam": ["KC", "KC"],
    })
    result = add_drive_touchdown_target(df)
    assert result["drive_td"].tolist() == [0, 0]


def test_engineer_features_creates_expected_columns():
    df = pd.DataFrame({
        "down": [1, 2, 3, 4],
        "ydstogo": [10, 5, 3, 5],
        "yardline_100": [75, 50, 20, 5],
        "posteam_score": [7, 7, 14, 14],
        "defteam_score": [0, 3, 3, 7],
        "half_seconds_remaining": [1800, 1500, 900, 100],
    })
    result = engineer_features(df)
    assert "score_diff" in result.columns
    assert "goal_to_go" in result.columns
    assert "red_zone" in result.columns
    assert result["score_diff"].tolist() == [7, 4, 11, 7]
    assert result["goal_to_go"].tolist() == [0, 0, 0, 1]
    assert result["red_zone"].tolist() == [0, 0, 1, 1]


def test_engineer_features_goal_to_go_logic():
    df = pd.DataFrame({
        "down": [1, 2],
        "ydstogo": [5, 10],
        "yardline_100": [5, 30],
        "posteam_score": [0, 0],
        "defteam_score": [0, 0],
        "half_seconds_remaining": [900, 900],
    })
    result = engineer_features(df)
    assert result["goal_to_go"].tolist() == [1, 0]


def test_split_by_season_separates_correctly():
    df = pd.DataFrame({
        "season": [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2015, 2016],
    })
    splits = split_by_season(df)
    assert len(splits["train"]) == 7
    assert len(splits["test"]) == 1
    assert len(splits["ood"]) == 2
    assert splits["test"]["season"].iloc[0] == 2025
    assert set(splits["ood"]["season"].unique()) == {2015, 2016}


def test_engineer_phase2_features_adds_team_quality():
    df = pd.DataFrame({
        "season": [2023, 2023, 2023, 2023],
        "posteam": ["KC", "KC", "BUF", "BUF"],
        "defteam": ["BUF", "BUF", "KC", "KC"],
        "epa": [0.3, 0.1, -0.1, 0.2],
        "posteam_type": ["home", "home", "away", "away"],
        "yardline_100": [5, 50, 5, 50],
        "touchdown": [1, 0, 0, 0],
        "td_team": ["KC", None, None, None],
    })
    result = engineer_phase2_features(df)
    assert "offense_epa" in result.columns
    assert "defense_rz_td_pct" in result.columns
    assert "is_home" in result.columns


def test_prepare_dataset_end_to_end():
    splits = prepare_dataset(seasons=list(range(2018, 2026)))
    train = splits["train"]
    assert len(train) > 50000
    for col in FEATURE_COLUMNS:
        assert col in train.columns, f"Missing feature column: {col}"
    assert TARGET_COLUMN in train.columns
    assert train[TARGET_COLUMN].isin([0, 1]).all()
    assert train["down"].between(1, 4).all()
    assert train["yardline_100"].between(1, 99).all()
