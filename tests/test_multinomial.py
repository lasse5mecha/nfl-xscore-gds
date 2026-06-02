import pandas as pd
import numpy as np
from src.data import add_drive_outcome_target, OUTCOME_CLASSES


def test_outcome_classes_constant():
    assert OUTCOME_CLASSES == ["punt_other", "td", "fg", "turnover"]


def test_td_drive_classified_correctly():
    df = pd.DataFrame({
        "game_id": ["G1"] * 3,
        "drive": [1, 1, 1],
        "posteam": ["KC"] * 3,
        "touchdown": [0, 0, 1],
        "td_team": [None, None, "KC"],
        "field_goal_result": [None, None, None],
        "interception": [0, 0, 0],
        "fumble_lost": [0, 0, 0],
        "fourth_down_failed": [0, 0, 0],
    })
    result = add_drive_outcome_target(df)
    assert result["drive_outcome_label"].tolist() == ["td", "td", "td"]


def test_fg_drive_classified_correctly():
    df = pd.DataFrame({
        "game_id": ["G1"] * 2,
        "drive": [1, 1],
        "posteam": ["KC"] * 2,
        "touchdown": [0, 0],
        "td_team": [None, None],
        "field_goal_result": [None, "made"],
        "interception": [0, 0],
        "fumble_lost": [0, 0],
        "fourth_down_failed": [0, 0],
    })
    result = add_drive_outcome_target(df)
    assert result["drive_outcome_label"].tolist() == ["fg", "fg"]


def test_turnover_drive_classified_correctly():
    df = pd.DataFrame({
        "game_id": ["G1"] * 2,
        "drive": [1, 1],
        "posteam": ["KC"] * 2,
        "touchdown": [0, 0],
        "td_team": [None, None],
        "field_goal_result": [None, None],
        "interception": [0, 1],
        "fumble_lost": [0, 0],
        "fourth_down_failed": [0, 0],
    })
    result = add_drive_outcome_target(df)
    assert result["drive_outcome_label"].tolist() == ["turnover", "turnover"]


def test_punt_drive_classified_correctly():
    df = pd.DataFrame({
        "game_id": ["G1"] * 2,
        "drive": [1, 1],
        "posteam": ["KC"] * 2,
        "touchdown": [0, 0],
        "td_team": [None, None],
        "field_goal_result": [None, None],
        "interception": [0, 0],
        "fumble_lost": [0, 0],
        "fourth_down_failed": [0, 0],
    })
    result = add_drive_outcome_target(df)
    assert result["drive_outcome_label"].tolist() == ["punt_other", "punt_other"]


def test_turnover_on_downs_classified_as_turnover():
    df = pd.DataFrame({
        "game_id": ["G1"] * 2,
        "drive": [1, 1],
        "posteam": ["KC"] * 2,
        "touchdown": [0, 0],
        "td_team": [None, None],
        "field_goal_result": [None, None],
        "interception": [0, 0],
        "fumble_lost": [0, 0],
        "fourth_down_failed": [0, 1],
    })
    result = add_drive_outcome_target(df)
    assert result["drive_outcome_label"].tolist() == ["turnover", "turnover"]


def test_drive_outcome_index_matches_classes():
    df = pd.DataFrame({
        "game_id": ["G1", "G1", "G1", "G1"],
        "drive": [1, 2, 3, 4],
        "posteam": ["KC"] * 4,
        "touchdown": [1, 0, 0, 0],
        "td_team": ["KC", None, None, None],
        "field_goal_result": [None, "made", None, None],
        "interception": [0, 0, 1, 0],
        "fumble_lost": [0, 0, 0, 0],
        "fourth_down_failed": [0, 0, 0, 0],
    })
    result = add_drive_outcome_target(df)
    labels = result.groupby("drive")["drive_outcome_label"].first().tolist()
    indices = result.groupby("drive")["drive_outcome"].first().tolist()
    assert labels == ["td", "fg", "turnover", "punt_other"]
    assert indices == [1, 2, 3, 0]


from src.model import (
    train_multinomial_model,
    train_calibrated_multinomial,
    predict_multinomial,
    evaluate_multinomial,
)


def _make_multinomial_data(n=2000):
    np.random.seed(42)
    X = pd.DataFrame({
        "down": np.random.choice([1, 2, 3, 4], n),
        "ydstogo": np.random.randint(1, 20, n),
        "yardline_100": np.random.randint(1, 99, n),
        "score_diff": np.random.randint(-14, 14, n),
        "half_seconds_remaining": np.random.randint(0, 1800, n),
        "goal_to_go": np.random.choice([0, 1], n),
        "red_zone": np.random.choice([0, 1], n),
    })
    probs = np.array([0.52, 0.18, 0.15, 0.15])
    y = np.random.choice(4, size=n, p=probs)
    return X, y


def test_train_multinomial_model_returns_fitted():
    X, y = _make_multinomial_data()
    model = train_multinomial_model(X, y)
    assert model is not None
    assert model.n_classes_ == 4


def test_predict_multinomial_returns_4_class_probabilities():
    X, y = _make_multinomial_data()
    model = train_multinomial_model(X, y)
    probs = predict_multinomial(model, X)
    assert probs.shape == (len(X), 4)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert (probs >= 0).all()
    assert (probs <= 1).all()


def test_train_calibrated_multinomial_returns_4_calibrators():
    X, y = _make_multinomial_data()
    model, calibrators = train_calibrated_multinomial(X, y)
    assert len(calibrators) == 4
    assert model.n_classes_ == 4


def test_predict_multinomial_with_calibration():
    X, y = _make_multinomial_data()
    model, calibrators = train_calibrated_multinomial(X, y, cv_folds=3)
    probs = predict_multinomial(model, X, calibrators=calibrators)
    assert probs.shape == (len(X), 4)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


from src.model import compute_xep_lookup_table, compute_xep, compute_drive_xvoa_ep, compute_gds_from_ep


def test_xep_lookup_table_structure():
    """Lookup table maps yardline (1-99) to expected points for opponent starting there."""
    np.random.seed(42)
    n_drives = 500
    lookup_data = pd.DataFrame({
        "start_yardline_100": np.random.randint(1, 100, n_drives),
        "actual_points": np.random.choice([0, 3, 7], n_drives, p=[0.6, 0.2, 0.2]),
    })
    table = compute_xep_lookup_table(lookup_data)
    assert len(table) > 0
    assert all(1 <= yl <= 99 for yl in table.keys())


def test_xep_from_probabilities():
    """xEP = P(TD)*7 + P(FG)*3 + P(TO)*(-opp_ep) + P(Punt)*0."""
    probs = np.array([[0.5, 0.2, 0.1, 0.2]])  # [punt, td, fg, turnover]
    turnover_yardlines = np.array([50])
    lookup = {yl: 2.0 for yl in range(1, 100)}
    xep = compute_xep(probs, turnover_yardlines, lookup)
    # 0.2*7 + 0.1*3 + 0.2*(-2.0) + 0.5*0 = 1.4 + 0.3 - 0.4 = 1.3
    assert abs(xep[0] - 1.3) < 0.01


def test_drive_xvoa_ep_positive_for_td():
    """TD drive should have positive xVOA (actual 7 minus xEP which is < 7)."""
    drives = pd.DataFrame({
        "game_id": ["G1", "G1"],
        "posteam": ["KC", "KC"],
        "drive": [1, 2],
        "actual_points": [7, 0],
        "xep": [2.5, 1.8],
    })
    result = compute_drive_xvoa_ep(drives)
    assert result.loc[result["drive"] == 1, "drive_xvoa"].iloc[0] == 4.5
    assert result.loc[result["drive"] == 2, "drive_xvoa"].iloc[0] == -1.8


def test_gds_from_ep_decomposition():
    """GDS = Off_xVOA + Def_xVOA (negated opponent off) + ST_Value."""
    drive_xvoa = pd.DataFrame({
        "game_id": ["G1", "G1", "G1", "G1"],
        "posteam": ["KC", "KC", "BUF", "BUF"],
        "drive": [1, 3, 2, 4],
        "drive_xvoa": [3.0, 1.0, -1.0, 2.0],
    })
    st_df = pd.DataFrame({
        "game_id": ["G1", "G1"],
        "posteam": ["KC", "BUF"],
        "st_value": [0.5, -0.3],
    })
    result = compute_gds_from_ep(drive_xvoa, st_df)
    kc = result[result["posteam"] == "KC"].iloc[0]
    # KC off_xvoa = 3.0 + 1.0 = 4.0
    # KC def_xvoa = -(BUF off_xvoa) = -(-1.0 + 2.0) = -1.0
    # KC gds = 4.0 + (-1.0) + 0.5 = 3.5
    assert abs(kc["offensive_xvoa"] - 4.0) < 0.01
    assert abs(kc["defensive_xvoa"] - (-1.0)) < 0.01
    assert abs(kc["gds"] - 3.5) < 0.01


def test_evaluate_multinomial_returns_per_class_metrics():
    X, y = _make_multinomial_data()
    model = train_multinomial_model(X, y)
    metrics = evaluate_multinomial(model, X, y)
    assert "multiclass_brier" in metrics
    assert "per_class_auc" in metrics
    assert len(metrics["per_class_auc"]) == 4
    assert all(0.0 <= v <= 1.0 for v in metrics["per_class_auc"].values())
    assert 0.0 <= metrics["multiclass_brier"] <= 2.0
