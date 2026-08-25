import numpy as np
import pandas as pd
from src.model import train_xscore_model, predict_xscore, evaluate_model, compute_game_xscore, compute_xvoa


def test_train_xscore_model_returns_fitted_model():
    np.random.seed(42)
    n = 1000
    X = pd.DataFrame({
        "down": np.random.choice([1, 2, 3, 4], n),
        "ydstogo": np.random.randint(1, 20, n),
        "yardline_100": np.random.randint(1, 99, n),
        "score_diff": np.random.randint(-14, 14, n),
        "half_seconds_remaining": np.random.randint(0, 1800, n),
        "goal_to_go": np.random.choice([0, 1], n),
        "red_zone": np.random.choice([0, 1], n),
    })
    y = (X["yardline_100"] < 20).astype(int)
    model = train_xscore_model(X, y)
    assert model is not None


def test_predict_xscore_returns_probabilities():
    np.random.seed(42)
    n = 1000
    X = pd.DataFrame({
        "down": np.random.choice([1, 2, 3, 4], n),
        "ydstogo": np.random.randint(1, 20, n),
        "yardline_100": np.random.randint(1, 99, n),
        "score_diff": np.random.randint(-14, 14, n),
        "half_seconds_remaining": np.random.randint(0, 1800, n),
        "goal_to_go": np.random.choice([0, 1], n),
        "red_zone": np.random.choice([0, 1], n),
    })
    y = (X["yardline_100"] < 20).astype(int)
    model = train_xscore_model(X, y)
    preds = predict_xscore(model, X)
    assert len(preds) == n
    assert all(0 <= p <= 1 for p in preds)


def test_evaluate_model_returns_metrics():
    np.random.seed(42)
    n = 500
    X = pd.DataFrame({
        "down": np.random.choice([1, 2, 3, 4], n),
        "ydstogo": np.random.randint(1, 20, n),
        "yardline_100": np.random.randint(1, 99, n),
        "score_diff": np.random.randint(-14, 14, n),
        "half_seconds_remaining": np.random.randint(0, 1800, n),
        "goal_to_go": np.random.choice([0, 1], n),
        "red_zone": np.random.choice([0, 1], n),
    })
    y = (X["yardline_100"] < 20).astype(int)
    model = train_xscore_model(X, y)
    metrics = evaluate_model(model, X, y)
    assert "brier_score" in metrics
    assert "auc_roc" in metrics
    assert "calibration" in metrics
    assert 0 <= metrics["brier_score"] <= 1
    assert 0 <= metrics["auc_roc"] <= 1
    assert isinstance(metrics["calibration"], dict)


def test_compute_game_xscore_sums_first_play_per_drive():
    df = pd.DataFrame({
        "game_id": ["G1", "G1", "G1", "G1", "G1"],
        "posteam": ["KC", "KC", "KC", "KC", "BUF"],
        "drive": [1, 1, 1, 2, 3],
        "xscore": [0.15, 0.25, 0.40, 0.30, 0.50],
        "play_order": [1, 2, 3, 4, 5],
    })
    result = compute_game_xscore(df)
    kc_row = result[result["posteam"] == "KC"].iloc[0]
    buf_row = result[result["posteam"] == "BUF"].iloc[0]
    assert abs(kc_row["game_xscore"] - 0.45) < 0.01  # 0.15 + 0.30
    assert abs(buf_row["game_xscore"] - 0.50) < 0.01


def test_compute_xvoa_measures_value_added():
    """xVOA sums play-to-play deltas; TD drive has positive total, failed drive negative."""
    df = pd.DataFrame({
        "game_id": ["G1", "G1", "G1", "G1", "G1"],
        "posteam": ["KC", "KC", "KC", "KC", "KC"],
        "drive": [1, 1, 1, 2, 2],
        "xscore": [0.20, 0.40, 0.70, 0.30, 0.15],
        "drive_td": [1, 1, 1, 0, 0],
        "half_seconds_remaining": [1800, 1750, 1700, 1600, 1550],
    })
    result = compute_xvoa(df)
    kc = result[result["posteam"] == "KC"].iloc[0]
    # Drive 1: deltas = (0.40-0.20) + (0.70-0.40) + (1.0-0.70) = 0.2+0.3+0.3 = 0.8
    # Drive 2: deltas = (0.15-0.30) + (0.0-0.15) = -0.15 + -0.15 = -0.30
    # Total xVOA = 0.8 + (-0.3) = 0.5
    assert abs(kc["xVOA"] - 0.5) < 0.01
