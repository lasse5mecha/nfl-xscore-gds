import os

import numpy as np
import pandas as pd
from src.model import train_calibrated_model, predict_xscore, save_calibrator, load_calibrator


def test_train_calibrated_model_returns_model_and_calibrator():
    """Calibrated training returns both model and isotonic calibrator."""
    np.random.seed(42)
    n = 2000
    X = pd.DataFrame({
        "down": np.random.choice([1, 2, 3, 4], n),
        "ydstogo": np.random.randint(1, 20, n),
        "yardline_100": np.random.randint(1, 99, n),
        "score_diff": np.random.randint(-14, 14, n),
        "half_seconds_remaining": np.random.randint(0, 1800, n),
        "goal_to_go": np.random.choice([0, 1], n),
        "red_zone": np.random.choice([0, 1], n),
        "rolling_offense_epa": np.random.normal(0, 0.05, n),
        "rolling_defense_epa": np.random.normal(0, 0.05, n),
        "momentum_epa": np.random.normal(0, 0.1, n),
        "is_home": np.random.choice([0, 1], n),
    })
    y = (X["yardline_100"] < 20).astype(int)
    model, calibrator = train_calibrated_model(X, y)
    assert model is not None
    assert calibrator is not None


def test_calibrated_predictions_are_probabilities():
    """Predictions with calibrator still produce valid probabilities."""
    np.random.seed(42)
    n = 2000
    X = pd.DataFrame({
        "down": np.random.choice([1, 2, 3, 4], n),
        "ydstogo": np.random.randint(1, 20, n),
        "yardline_100": np.random.randint(1, 99, n),
        "score_diff": np.random.randint(-14, 14, n),
        "half_seconds_remaining": np.random.randint(0, 1800, n),
        "goal_to_go": np.random.choice([0, 1], n),
        "red_zone": np.random.choice([0, 1], n),
        "rolling_offense_epa": np.random.normal(0, 0.05, n),
        "rolling_defense_epa": np.random.normal(0, 0.05, n),
        "momentum_epa": np.random.normal(0, 0.1, n),
        "is_home": np.random.choice([0, 1], n),
    })
    y = (X["yardline_100"] < 20).astype(int)
    model, calibrator = train_calibrated_model(X, y)
    preds = predict_xscore(model, X, calibrator=calibrator)
    assert len(preds) == n
    assert all(0 <= p <= 1 for p in preds)


def test_calibrator_changes_predictions():
    """Calibrator should modify raw predictions (not pass-through)."""
    np.random.seed(42)
    n = 2000
    X = pd.DataFrame({
        "down": np.random.choice([1, 2, 3, 4], n),
        "ydstogo": np.random.randint(1, 20, n),
        "yardline_100": np.random.randint(1, 99, n),
        "score_diff": np.random.randint(-14, 14, n),
        "half_seconds_remaining": np.random.randint(0, 1800, n),
        "goal_to_go": np.random.choice([0, 1], n),
        "red_zone": np.random.choice([0, 1], n),
        "rolling_offense_epa": np.random.normal(0, 0.05, n),
        "rolling_defense_epa": np.random.normal(0, 0.05, n),
        "momentum_epa": np.random.normal(0, 0.1, n),
        "is_home": np.random.choice([0, 1], n),
    })
    y = (X["yardline_100"] < 20).astype(int)
    model, calibrator = train_calibrated_model(X, y)
    raw_preds = predict_xscore(model, X)
    cal_preds = predict_xscore(model, X, calibrator=calibrator)
    # They should differ (isotonic mapping won't be identity)
    assert not np.allclose(raw_preds, cal_preds)


def test_save_and_load_calibrator(tmp_path):
    """Calibrator can be saved to disk and loaded back."""
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
        "rolling_offense_epa": np.random.normal(0, 0.05, n),
        "rolling_defense_epa": np.random.normal(0, 0.05, n),
        "momentum_epa": np.random.normal(0, 0.1, n),
        "is_home": np.random.choice([0, 1], n),
    })
    y = (X["yardline_100"] < 20).astype(int)
    model, calibrator = train_calibrated_model(X, y)

    path = str(tmp_path / "calibrator.pkl")
    save_calibrator(calibrator, path)
    assert os.path.exists(path)

    loaded = load_calibrator(path)
    test_input = np.array([0.1, 0.5, 0.9])
    np.testing.assert_array_almost_equal(
        calibrator.predict(test_input),
        loaded.predict(test_input),
    )
