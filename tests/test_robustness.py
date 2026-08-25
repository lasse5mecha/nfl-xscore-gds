import pandas as pd
import numpy as np
from src.robustness import compute_opponent_adjustment
from src.robustness import compute_field_position_mediation
from src.robustness import run_controlled_regressions


def test_opponent_adjustment_excludes_focal_game():
    """Leave-one-out: opponent GDS excludes the game against the focal team."""
    season_gds = pd.DataFrame({
        "season": [2023] * 6,
        "team": ["KC", "KC", "BUF", "BUF", "MIA", "MIA"],
        "game_id": ["G1", "G2", "G1", "G3", "G2", "G3"],
        "opponent": ["BUF", "MIA", "KC", "MIA", "KC", "BUF"],
        "gds": [5.0, 3.0, 2.0, 4.0, -1.0, 1.0],
    })
    result = compute_opponent_adjustment(season_gds)
    # KC played BUF (G1) and MIA (G2)
    # BUF's GDS excluding G1 (vs KC): only G3 → gds=4.0
    # MIA's GDS excluding G2 (vs KC): only G3 → gds=1.0
    # KC's opp_avg_gds = mean(4.0, 1.0) = 2.5
    kc = result[(result["team"] == "KC") & (result["season"] == 2023)].iloc[0]
    assert abs(kc["opp_avg_gds"] - 2.5) < 0.01


def test_opponent_adjustment_output_columns():
    season_gds = pd.DataFrame({
        "season": [2023, 2023],
        "team": ["KC", "BUF"],
        "game_id": ["G1", "G1"],
        "opponent": ["BUF", "KC"],
        "gds": [5.0, 2.0],
    })
    result = compute_opponent_adjustment(season_gds)
    assert "opp_avg_gds" in result.columns
    assert "season" in result.columns
    assert "team" in result.columns


def test_mediation_analysis_returns_expected_keys():
    """Mediation analysis should return correlation coefficients and significance."""
    np.random.seed(42)
    n = 30
    team_seasons = pd.DataFrame({
        "season": [2023] * n,
        "team": [f"T{i}" for i in range(n)],
        "off_xvoa_per_game": np.random.randn(n),
        "def_xvoa_per_game": np.random.randn(n),
        "avg_start_yardline": np.random.uniform(25, 40, n),
        "playoff_wins": np.random.randint(0, 4, n),
    })
    result = compute_field_position_mediation(team_seasons)
    assert "def_to_field_pos_r" in result
    assert "def_to_field_pos_p" in result
    assert "off_xvoa_to_wins_r" in result
    assert "off_xvoa_to_wins_partial_r" in result
    assert "mediation_pct" in result


def test_mediation_with_no_mediation_returns_low_pct():
    """When defense and field position are uncorrelated, mediation should be ~0."""
    np.random.seed(42)
    n = 50
    team_seasons = pd.DataFrame({
        "season": [2023] * n,
        "team": [f"T{i}" for i in range(n)],
        "off_xvoa_per_game": np.random.randn(n) * 2,
        "def_xvoa_per_game": np.random.randn(n),
        "avg_start_yardline": np.random.uniform(25, 40, n),
        "playoff_wins": np.zeros(n),
    })
    team_seasons["playoff_wins"] = (team_seasons["off_xvoa_per_game"] * 0.5 + np.random.randn(n) * 0.3).clip(0, 4).round()
    result = compute_field_position_mediation(team_seasons)
    assert result["mediation_pct"] < 50


def test_controlled_regressions_output():
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "off_xvoa_per_game": np.random.randn(n),
        "def_xvoa_per_game": np.random.randn(n),
        "opp_avg_gds": np.random.randn(n) * 0.5,
        "made_playoffs": np.random.choice([0, 1], n),
        "playoff_wins": np.random.randint(0, 4, n),
    })
    result = run_controlled_regressions(df)
    assert "logit_uncontrolled" in result
    assert "logit_controlled" in result
    assert "ols_uncontrolled" in result
    assert "ols_controlled" in result
    assert "off_coef" in result["logit_controlled"]
    assert "off_pvalue" in result["logit_controlled"]
    assert "opp_coef" in result["logit_controlled"]
