import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
from src.viz import plot_field_heatmap, plot_calibration, plot_shap_summary, plot_team_performance, plot_interactive_heatmap, plot_coach_aggressiveness


def test_plot_field_heatmap_creates_file(tmp_path):
    predictions = pd.DataFrame({
        "down": [1, 2, 3, 4] * 25,
        "yardline_100": list(range(1, 26)) * 4,
        "xscore": np.random.uniform(0, 1, 100),
    })
    output_path = tmp_path / "heatmap.png"
    plot_field_heatmap(predictions, str(output_path))
    assert output_path.exists()
    assert output_path.stat().st_size > 1000


def test_plot_calibration_creates_file(tmp_path):
    metrics = {
        "calibration": {
            "prob_true": [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
            "prob_pred": [0.05, 0.14, 0.26, 0.34, 0.46, 0.54, 0.66, 0.74, 0.86, 0.94],
        },
        "brier_score": 0.12,
        "auc_roc": 0.85,
    }
    output_path = tmp_path / "calibration.png"
    plot_calibration(metrics, str(output_path))
    assert output_path.exists()
    assert output_path.stat().st_size > 1000


def test_plot_shap_summary_creates_file(tmp_path):
    np.random.seed(42)
    n = 200
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
    model = xgb.XGBClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    output_path = tmp_path / "shap.png"
    plot_shap_summary(model, X, str(output_path))
    assert output_path.exists()


def test_plot_team_performance_creates_file(tmp_path):
    df = pd.DataFrame({
        "team": ["KC", "BUF", "SF", "DAL"],
        "expected_tds": [3.5, 3.0, 2.8, 2.0],
        "actual_tds": [4.0, 2.5, 3.5, 1.8],
    })
    output_path = tmp_path / "teams.png"
    plot_team_performance(df, str(output_path))
    assert output_path.exists()


def test_plot_interactive_heatmap_creates_html(tmp_path):
    predictions = pd.DataFrame({
        "down": [1, 2, 3, 4] * 25,
        "yardline_100": list(range(1, 26)) * 4,
        "xscore": np.random.uniform(0, 1, 100),
    })
    output_path = tmp_path / "heatmap.html"
    plot_interactive_heatmap(predictions, str(output_path))
    assert output_path.exists()
    assert output_path.stat().st_size > 1000
    content = output_path.read_text()
    assert "plotly" in content.lower()


def test_plot_coach_aggressiveness_creates_file(tmp_path):
    df = pd.DataFrame({
        "coach": ["Reid", "Belichick", "McDermott", "Shanahan"],
        "pct_optimal": [0.82, 0.65, 0.73, 0.78],
        "total_4th_downs": [45, 38, 42, 40],
    })
    output_path = tmp_path / "coaches.png"
    plot_coach_aggressiveness(df, str(output_path))
    assert output_path.exists()
