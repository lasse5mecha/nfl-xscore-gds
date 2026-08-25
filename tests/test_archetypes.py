import pandas as pd
import numpy as np
from src.data import compute_playoff_outcomes


def test_playoff_outcomes_identifies_playoff_teams():
    """Teams that appear in POST games are playoff teams."""
    raw_df = pd.DataFrame({
        "season": [2023] * 6,
        "season_type": ["REG", "REG", "POST", "POST", "POST", "POST"],
        "week": [1, 1, 19, 19, 20, 20],
        "game_id": ["G1", "G1", "G2", "G2", "G3", "G3"],
        "home_team": ["KC", "BUF", "KC", "MIA", "KC", "BUF"],
        "away_team": ["MIA", "NYJ", "MIA", "BUF", "BUF", "NYJ"],
        "home_score": [24, 30, 26, 14, 27, 20],
        "away_score": [17, 20, 7, 21, 24, 17],
    })
    result = compute_playoff_outcomes(raw_df)
    playoff_teams = set(result[result["made_playoffs"]]["team"])
    assert "KC" in playoff_teams
    assert "BUF" in playoff_teams
    assert "MIA" in playoff_teams
    assert "NYJ" not in playoff_teams  # NYJ only in REG


def test_playoff_wins_counts_actual_wins():
    """Playoff wins = games won in postseason, not rounds reached."""
    raw_df = pd.DataFrame({
        "season": [2023] * 4,
        "season_type": ["POST", "POST", "POST", "POST"],
        "week": [19, 19, 20, 20],
        "game_id": ["G1", "G1", "G2", "G2"],
        "home_team": ["KC", "KC", "KC", "KC"],
        "away_team": ["MIA", "MIA", "BUF", "BUF"],
        "home_score": [26, 26, 27, 27],
        "away_score": [7, 7, 24, 24],
    })
    result = compute_playoff_outcomes(raw_df)
    kc = result[(result["team"] == "KC") & (result["season"] == 2023)].iloc[0]
    assert kc["playoff_wins"] == 2  # Won WC + Won Div
    mia = result[(result["team"] == "MIA") & (result["season"] == 2023)].iloc[0]
    assert mia["playoff_wins"] == 0  # Lost in WC
    buf = result[(result["team"] == "BUF") & (result["season"] == 2023)].iloc[0]
    assert buf["playoff_wins"] == 0  # Lost in Div


def test_super_bowl_winner_identified():
    """Team that wins week 22 (Super Bowl) gets won_super_bowl=True."""
    raw_df = pd.DataFrame({
        "season": [2023] * 2,
        "season_type": ["POST", "POST"],
        "week": [22, 22],
        "game_id": ["SB", "SB"],
        "home_team": ["KC", "KC"],
        "away_team": ["SF", "SF"],
        "home_score": [25, 25],
        "away_score": [22, 22],
    })
    result = compute_playoff_outcomes(raw_df)
    kc = result[(result["team"] == "KC") & (result["season"] == 2023)].iloc[0]
    sf = result[(result["team"] == "SF") & (result["season"] == 2023)].iloc[0]
    assert kc["won_super_bowl"] == True
    assert sf["won_super_bowl"] == False


def test_regular_season_wins_computed():
    """Regular season wins counted from REG games only."""
    raw_df = pd.DataFrame({
        "season": [2023] * 6,
        "season_type": ["REG", "REG", "REG", "REG", "POST", "POST"],
        "week": [1, 1, 2, 2, 19, 19],
        "game_id": ["G1", "G1", "G2", "G2", "G3", "G3"],
        "home_team": ["KC", "BUF", "KC", "BUF", "KC", "KC"],
        "away_team": ["BUF", "KC", "MIA", "MIA", "BUF", "BUF"],
        "home_score": [27, 20, 31, 24, 30, 30],
        "away_score": [24, 24, 17, 21, 20, 20],
    })
    result = compute_playoff_outcomes(raw_df)
    kc = result[(result["team"] == "KC") & (result["season"] == 2023)].iloc[0]
    # KC won game G1 (home 27-24) and G2 (home 31-17) = 2 reg wins
    assert kc["reg_wins"] == 2
