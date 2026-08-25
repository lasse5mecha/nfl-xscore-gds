import time

import numpy as np
import pandas as pd
from src.data import compute_rolling_epa, compute_momentum_epa
from src.data import engineer_phase2_rolling_features, PHASE2_FEATURE_COLUMNS
from src.data import prepare_phase2_dataset, TARGET_COLUMN


def test_compute_rolling_epa_basic():
    """Rolling EPA for a team across 3 games in a season."""
    df = pd.DataFrame({
        "season": [2023] * 9,
        "week": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "game_id": ["G1", "G1", "G1", "G2", "G2", "G2", "G3", "G3", "G3"],
        "posteam": ["KC"] * 9,
        "defteam": ["BUF"] * 3 + ["LV"] * 3 + ["DEN"] * 3,
        "epa": [0.3, 0.1, 0.2, -0.1, 0.0, 0.1, 0.4, 0.2, 0.3],
    })
    result = compute_rolling_epa(df)
    # Week 1: no prior games → Bayesian prior → 0.0
    assert all(result.loc[result["week"] == 1, "rolling_offense_epa"] == 0.0)
    # Week 2: 1 prior game (avg EPA = 0.2), Bayesian: (1*0.2 + 4*0.0)/(1+4) = 0.04
    week2_vals = result.loc[result["week"] == 2, "rolling_offense_epa"]
    assert np.isclose(week2_vals.iloc[0], 0.04, atol=0.001)


def test_rolling_epa_season_reset():
    """EPA resets to prior at the start of each season."""
    df = pd.DataFrame({
        "season": [2022, 2022, 2023, 2023],
        "week": [17, 17, 1, 1],
        "game_id": ["G1", "G1", "G2", "G2"],
        "posteam": ["KC", "KC", "KC", "KC"],
        "defteam": ["BUF", "BUF", "BUF", "BUF"],
        "epa": [0.5, 0.6, 0.1, 0.2],
    })
    result = compute_rolling_epa(df)
    # 2023 week 1 should not use 2022 data → rolling = 0.0 (prior)
    week1_2023 = result[(result["season"] == 2023) & (result["week"] == 1)]
    assert all(week1_2023["rolling_offense_epa"] == 0.0)


def test_rolling_defense_epa():
    """Defense rolling EPA tracks EPA allowed by defteam."""
    df = pd.DataFrame({
        "season": [2023] * 6,
        "week": [1, 1, 2, 2, 3, 3],
        "game_id": ["G1", "G1", "G2", "G2", "G3", "G3"],
        "posteam": ["KC", "KC", "BUF", "BUF", "KC", "KC"],
        "defteam": ["BUF", "BUF", "KC", "KC", "BUF", "BUF"],
        "epa": [0.4, 0.2, -0.1, 0.1, 0.3, 0.1],
    })
    result = compute_rolling_epa(df)
    # Week 3, defteam=BUF: BUF allowed avg EPA=0.3 in Week 1
    # Bayesian: (1*0.3 + 4*0.0) / (1+4) = 0.06
    week3 = result[(result["week"] == 3) & (result["defteam"] == "BUF")]
    assert np.isclose(week3["rolling_defense_epa"].iloc[0], 0.06, atol=0.001)


def test_momentum_epa_basic():
    """Momentum is mean EPA over last 10 offensive plays by posteam in game."""
    df = pd.DataFrame({
        "game_id": ["G1"] * 12,
        "posteam": ["KC"] * 12,
        "play_type": ["pass"] * 12,
        "epa": [0.1, 0.2, 0.3, -0.1, 0.4, 0.0, 0.2, -0.2, 0.5, 0.1, 0.3, -0.1],
    })
    result = compute_momentum_epa(df)
    # First play: no prior plays → 0.0
    assert result["momentum_epa"].iloc[0] == 0.0
    # Second play: mean of [0.1] = 0.1
    assert np.isclose(result["momentum_epa"].iloc[1], 0.1)
    # 11th play: mean of last 10 plays (indices 0-9 EPAs)
    last_10 = [0.1, 0.2, 0.3, -0.1, 0.4, 0.0, 0.2, -0.2, 0.5, 0.1]
    assert np.isclose(result["momentum_epa"].iloc[10], np.mean(last_10), atol=0.001)


def test_momentum_epa_first_play_neutral():
    """First offensive play of a game gets momentum = 0.0."""
    df = pd.DataFrame({
        "game_id": ["G1", "G1", "G2"],
        "posteam": ["KC", "KC", "KC"],
        "play_type": ["pass", "run", "pass"],
        "epa": [0.5, 0.3, 0.2],
    })
    result = compute_momentum_epa(df)
    assert result["momentum_epa"].iloc[0] == 0.0
    # First play of G2 also gets 0.0 (different game)
    assert result["momentum_epa"].iloc[2] == 0.0


def test_momentum_epa_separate_teams():
    """Momentum is computed per-team within a game."""
    df = pd.DataFrame({
        "game_id": ["G1"] * 4,
        "posteam": ["KC", "KC", "BUF", "KC"],
        "play_type": ["pass", "run", "pass", "pass"],
        "epa": [0.5, 0.3, 0.8, 0.1],
    })
    result = compute_momentum_epa(df)
    # 4th play (KC, index 3): prior KC plays in G1 are [0.5, 0.3] → mean = 0.4
    assert np.isclose(result["momentum_epa"].iloc[3], 0.4)


def test_engineer_phase2_rolling_features_adds_all_columns():
    """Integration: all Phase 2 columns are present after engineering."""
    df = pd.DataFrame({
        "season": [2023] * 6,
        "week": [1, 1, 1, 2, 2, 2],
        "game_id": ["G1", "G1", "G1", "G2", "G2", "G2"],
        "posteam": ["KC", "KC", "KC", "KC", "KC", "KC"],
        "defteam": ["BUF", "BUF", "BUF", "LV", "LV", "LV"],
        "epa": [0.3, 0.1, 0.2, -0.1, 0.4, 0.2],
        "play_type": ["pass", "run", "pass", "pass", "run", "pass"],
        "posteam_type": ["home", "home", "home", "away", "away", "away"],
        "down": [1, 2, 3, 1, 2, 3],
        "ydstogo": [10, 7, 3, 10, 5, 2],
        "yardline_100": [75, 68, 65, 50, 45, 43],
        "posteam_score": [0, 0, 0, 7, 7, 7],
        "defteam_score": [0, 0, 0, 0, 0, 0],
        "half_seconds_remaining": [1800, 1750, 1700, 1800, 1760, 1720],
    })
    result = engineer_phase2_rolling_features(df)
    expected_cols = ["rolling_offense_epa", "rolling_defense_epa", "momentum_epa", "is_home"]
    for col in expected_cols:
        assert col in result.columns, f"Missing: {col}"
    # is_home should be 1 for week 1 (home), 0 for week 2 (away)
    assert all(result.loc[result["week"] == 1, "is_home"] == 1)
    assert all(result.loc[result["week"] == 2, "is_home"] == 0)
    # No NaN in new features
    for col in expected_cols:
        assert result[col].notna().all(), f"NaN found in {col}"


def test_prepare_phase2_dataset_structure():
    """prepare_phase2_dataset returns splits with all Phase 2 feature columns."""
    splits = prepare_phase2_dataset(seasons=[2023, 2024])
    train = splits["train"]
    assert len(train) > 10000
    for col in PHASE2_FEATURE_COLUMNS:
        assert col in train.columns, f"Missing: {col}"
    assert TARGET_COLUMN in train.columns
    assert train["rolling_offense_epa"].notna().all()
    assert train["rolling_defense_epa"].notna().all()
    assert train["momentum_epa"].notna().all()


def test_momentum_epa_performance():
    """Momentum computation should handle 50K plays in under 10 seconds."""
    np.random.seed(42)
    n = 50000
    df = pd.DataFrame({
        "game_id": [f"G{i // 100}" for i in range(n)],
        "posteam": np.random.choice(["KC", "BUF", "SF", "PHI"], n),
        "play_type": np.random.choice(["pass", "run"], n),
        "epa": np.random.normal(0, 0.3, n),
    })
    start = time.time()
    result = compute_momentum_epa(df)
    elapsed = time.time() - start
    assert elapsed < 10.0, f"Too slow: {elapsed:.1f}s"
    assert len(result) == n
