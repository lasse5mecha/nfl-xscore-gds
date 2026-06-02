import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_predict


def train_xscore_model(
    X: pd.DataFrame,
    y: "pd.Series | np.ndarray",
    params: "dict | None" = None,
) -> xgb.XGBClassifier:
    """Train XGBoost classifier for xScore prediction."""
    if params is None:
        params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "random_state": 42,
        }
    model = xgb.XGBClassifier(**params)
    model.fit(X, y)
    return model


def train_calibrated_model(
    X: pd.DataFrame,
    y: "pd.Series | np.ndarray",
    params: "dict | None" = None,
    cv_folds: int = 5,
) -> tuple[xgb.XGBClassifier, IsotonicRegression]:
    """Train XGBoost with isotonic calibration via cross-validated OOF predictions."""
    if params is None:
        params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "random_state": 42,
        }
    base_model = xgb.XGBClassifier(**params)
    oof_preds = cross_val_predict(base_model, X, y, cv=cv_folds, method="predict_proba")[:, 1]

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof_preds, y)

    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y)

    return final_model, calibrator


def predict_xscore(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    calibrator: "IsotonicRegression | None" = None,
) -> np.ndarray:
    """Predict touchdown probability for each play."""
    raw_preds = model.predict_proba(X)[:, 1]
    if calibrator is not None:
        return calibrator.predict(raw_preds)
    return raw_preds


def save_calibrator(calibrator: IsotonicRegression, path: str = "models/xscore_v2_calibrator.pkl") -> None:
    """Save isotonic calibrator to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, path)


def load_calibrator(path: str = "models/xscore_v2_calibrator.pkl") -> IsotonicRegression:
    """Load isotonic calibrator from disk."""
    return joblib.load(path)


def save_model(model: xgb.XGBClassifier, path: str = "models/xscore_v1.json") -> None:
    """Save model to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)


def load_model(path: str = "models/xscore_v1.json") -> xgb.XGBClassifier:
    """Load model from disk."""
    model = xgb.XGBClassifier()
    model.load_model(path)
    return model


def evaluate_model(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    y: "pd.Series | np.ndarray",
) -> dict:
    """Compute Brier score, AUC-ROC, and calibration data."""
    preds = predict_xscore(model, X)
    brier = brier_score_loss(y, preds)
    auc = roc_auc_score(y, preds)
    prob_true, prob_pred = calibration_curve(y, preds, n_bins=10)
    calibration = {
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
    }
    return {
        "brier_score": brier,
        "auc_roc": auc,
        "calibration": calibration,
    }


def compute_game_xscore(df: pd.DataFrame) -> pd.DataFrame:
    """Sum xScore at the first play of each drive to get per-team game xScore."""
    first_plays = df.groupby(["game_id", "posteam", "drive"]).first().reset_index()
    game_xscore = (
        first_plays.groupby(["game_id", "posteam"])["xscore"]
        .sum()
        .reset_index()
        .rename(columns={"xscore": "game_xscore"})
    )
    return game_xscore


def compute_xvoa(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Offensive Value Added (xVOA) per team per game.

    xVOA = sum of play-to-play xScore deltas within each drive.
    Measures how much the offense improved TD probability beyond expectation.
    On the last play of a drive, delta goes to 1.0 (TD) or 0.0 (no TD).
    """
    df = df.sort_values(
        ["game_id", "posteam", "drive", "half_seconds_remaining"],
        ascending=[True, True, True, False],
    ).copy()

    df["_xscore_next"] = df.groupby(["game_id", "posteam", "drive"])["xscore"].shift(-1)
    last_play = df["_xscore_next"].isna()
    df.loc[last_play & (df["drive_td"] == 1), "_xscore_next"] = 1.0
    df.loc[last_play & (df["drive_td"] == 0), "_xscore_next"] = 0.0
    df["xscore_delta"] = df["_xscore_next"] - df["xscore"]

    game_xvoa = (
        df.groupby(["game_id", "posteam"])["xscore_delta"]
        .sum()
        .reset_index()
        .rename(columns={"xscore_delta": "xVOA"})
    )
    df = df.drop(columns=["_xscore_next", "xscore_delta"])
    return game_xvoa


def compute_game_deserved_score(xvoa_df: pd.DataFrame, st_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Game Deserved Score per team per game.

    GDS = Offensive xVOA + Defensive xVOA + ST Value
    where Defensive xVOA = -(opponent's offensive xVOA in same game)
    """
    result = xvoa_df.copy().rename(columns={"xVOA": "offensive_xvoa"})

    def _get_opponent_xvoa(row, xvoa_df):
        game_rows = xvoa_df[xvoa_df["game_id"] == row["game_id"]]
        opponent = game_rows[game_rows["posteam"] != row["posteam"]]
        if len(opponent) == 0:
            return 0.0
        return -opponent["xVOA"].iloc[0]

    result["defensive_xvoa"] = result.apply(lambda r: _get_opponent_xvoa(r, xvoa_df), axis=1)

    result = result.merge(st_df[["game_id", "posteam", "st_value"]], on=["game_id", "posteam"], how="left")
    result["st_value"] = result["st_value"].fillna(0.0)

    result["gds"] = result["offensive_xvoa"] + result["defensive_xvoa"] + result["st_value"]

    return result[["game_id", "posteam", "offensive_xvoa", "defensive_xvoa", "st_value", "gds"]]


def train_multinomial_model(
    X: pd.DataFrame,
    y: "pd.Series | np.ndarray",
    params: "dict | None" = None,
) -> xgb.XGBClassifier:
    """Train 4-class multinomial XGBoost for drive outcome prediction."""
    if params is None:
        params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "multi:softprob",
            "num_class": 4,
            "eval_metric": "mlogloss",
            "random_state": 42,
        }
    model = xgb.XGBClassifier(**params)
    model.fit(X, y)
    return model


def train_calibrated_multinomial(
    X: pd.DataFrame,
    y: "pd.Series | np.ndarray",
    params: "dict | None" = None,
    cv_folds: int = 5,
) -> tuple[xgb.XGBClassifier, list[IsotonicRegression]]:
    """Train multinomial XGBoost with per-class isotonic calibration."""
    if params is None:
        params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "multi:softprob",
            "num_class": 4,
            "eval_metric": "mlogloss",
            "random_state": 42,
        }
    base_model = xgb.XGBClassifier(**params)
    oof_preds = cross_val_predict(base_model, X, y, cv=cv_folds, method="predict_proba")

    calibrators = []
    for cls in range(4):
        binary_target = (np.asarray(y) == cls).astype(int)
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(oof_preds[:, cls], binary_target)
        calibrators.append(cal)

    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y)

    return final_model, calibrators


def predict_multinomial(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    calibrators: "list[IsotonicRegression] | None" = None,
) -> np.ndarray:
    """Predict 4-class probabilities, optionally with per-class calibration."""
    raw_probs = model.predict_proba(X)
    if calibrators is None:
        return raw_probs
    calibrated = np.column_stack([
        calibrators[cls].predict(raw_probs[:, cls]) for cls in range(4)
    ])
    # Renormalize rows to sum to 1
    row_sums = calibrated.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return calibrated / row_sums


def check_intuition(model: xgb.XGBClassifier) -> dict[str, float]:
    """Sanity-check model predictions against expected values."""
    scenarios = pd.DataFrame([
        {"down": 1, "ydstogo": 1, "yardline_100": 1, "score_diff": 0,
         "half_seconds_remaining": 900, "goal_to_go": 1, "red_zone": 1},
        {"down": 1, "ydstogo": 5, "yardline_100": 5, "score_diff": 0,
         "half_seconds_remaining": 900, "goal_to_go": 1, "red_zone": 1},
        {"down": 1, "ydstogo": 10, "yardline_100": 70, "score_diff": 0,
         "half_seconds_remaining": 900, "goal_to_go": 0, "red_zone": 0},
        {"down": 4, "ydstogo": 22, "yardline_100": 50, "score_diff": 0,
         "half_seconds_remaining": 900, "goal_to_go": 0, "red_zone": 0},
    ])
    preds = predict_xscore(model, scenarios)
    return {
        "1st_goal_from_1": preds[0],
        "1st_goal_from_5": preds[1],
        "1st_10_own_30": preds[2],
        "4th_22_midfield": preds[3],
    }


def compute_xep_lookup_table(drive_data: pd.DataFrame) -> dict[int, float]:
    """Build lookup: for each starting yardline, mean actual points scored.

    Input DataFrame must have columns: start_yardline_100, actual_points.
    Returns dict mapping yardline (1-99) to average EP from that field position.
    """
    grouped = drive_data.groupby("start_yardline_100")["actual_points"].mean()
    return {int(yl): float(ep) for yl, ep in grouped.items() if 1 <= yl <= 99}


def compute_xep(
    probs: np.ndarray,
    turnover_yardlines: np.ndarray,
    opp_ep_lookup: dict[int, float],
) -> np.ndarray:
    """Compute expected points from multinomial probabilities.

    probs: shape (n, 4) with columns [punt_other, td, fg, turnover]
    turnover_yardlines: shape (n,) — yardline where opponent would start after TO
    opp_ep_lookup: maps yardline to opponent's expected points from there

    xEP = P(TD)*7 + P(FG)*3 + P(TO)*(-opp_EP) + P(Punt)*0
    """
    opp_ep = np.array([
        opp_ep_lookup.get(int(yl), 0.0) for yl in turnover_yardlines
    ])
    xep = (
        probs[:, 1] * 7.0
        + probs[:, 2] * 3.0
        + probs[:, 3] * (-opp_ep)
        + probs[:, 0] * 0.0
    )
    return xep


def compute_drive_xvoa_ep(drives: pd.DataFrame) -> pd.DataFrame:
    """Compute per-drive xVOA = actual_points - xEP."""
    drives = drives.copy()
    drives["drive_xvoa"] = drives["actual_points"] - drives["xep"]
    return drives


def compute_gds_from_ep(drive_xvoa: pd.DataFrame, st_df: pd.DataFrame) -> pd.DataFrame:
    """Compute GDS from drive-level expected-points xVOA.

    Off_xVOA = sum of drive_xvoa for team's drives
    Def_xVOA = -(sum of drive_xvoa for opponent's drives)
    GDS = Off_xVOA + Def_xVOA + ST_Value
    """
    off_xvoa = (
        drive_xvoa.groupby(["game_id", "posteam"])["drive_xvoa"]
        .sum()
        .reset_index()
        .rename(columns={"drive_xvoa": "offensive_xvoa"})
    )

    def _get_opponent_off_xvoa(row, off_df):
        game_rows = off_df[off_df["game_id"] == row["game_id"]]
        opponent = game_rows[game_rows["posteam"] != row["posteam"]]
        if len(opponent) == 0:
            return 0.0
        return -opponent["offensive_xvoa"].iloc[0]

    result = off_xvoa.copy()
    result["defensive_xvoa"] = result.apply(lambda r: _get_opponent_off_xvoa(r, off_xvoa), axis=1)
    result = result.merge(
        st_df[["game_id", "posteam", "st_value"]], on=["game_id", "posteam"], how="left"
    )
    result["st_value"] = result["st_value"].fillna(0.0)
    result["gds"] = result["offensive_xvoa"] + result["defensive_xvoa"] + result["st_value"]

    return result[["game_id", "posteam", "offensive_xvoa", "defensive_xvoa", "st_value", "gds"]]


def evaluate_multinomial(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    y: "pd.Series | np.ndarray",
    calibrators: "list[IsotonicRegression] | None" = None,
) -> dict:
    """Compute per-class AUC, multiclass Brier score, and calibration data."""
    probs = predict_multinomial(model, X, calibrators=calibrators)
    y_arr = np.asarray(y)
    n_classes = 4

    # Per-class AUC (one-vs-rest)
    per_class_auc = {}
    class_names = ["punt_other", "td", "fg", "turnover"]
    for cls in range(n_classes):
        binary_y = (y_arr == cls).astype(int)
        if binary_y.sum() > 0 and binary_y.sum() < len(binary_y):
            per_class_auc[class_names[cls]] = roc_auc_score(binary_y, probs[:, cls])
        else:
            per_class_auc[class_names[cls]] = float("nan")

    # Multiclass Brier score (mean of per-class Brier scores)
    brier_per_class = []
    for cls in range(n_classes):
        binary_y = (y_arr == cls).astype(int)
        brier_per_class.append(brier_score_loss(binary_y, probs[:, cls]))
    multiclass_brier = np.mean(brier_per_class)

    # Per-class calibration curves
    calibration_data = {}
    for cls in range(n_classes):
        binary_y = (y_arr == cls).astype(int)
        prob_true, prob_pred = calibration_curve(binary_y, probs[:, cls], n_bins=10)
        calibration_data[class_names[cls]] = {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        }

    return {
        "multiclass_brier": multiclass_brier,
        "per_class_auc": per_class_auc,
        "per_class_brier": {class_names[i]: brier_per_class[i] for i in range(n_classes)},
        "calibration": calibration_data,
    }
