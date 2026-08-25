import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost as xgb
import plotly.graph_objects as go
from pathlib import Path


def plot_field_heatmap(predictions: pd.DataFrame, output_path: str) -> None:
    """Create field position heatmap: X=yard line, Y=down, Color=xScore."""
    pivot = predictions.pivot_table(
        values="xscore", index="down", columns="yardline_100", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(16, 4))
    sns.heatmap(
        pivot,
        cmap="RdYlGn_r",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "xScore (TD Probability)"},
    )
    ax.set_xlabel("Yards from End Zone")
    ax.set_ylabel("Down")
    ax.set_title("xScore: Touchdown Probability by Field Position and Down")
    ax.invert_xaxis()
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_shap_summary(model: xgb.XGBClassifier, X: pd.DataFrame, output_path: str) -> None:
    """Generate SHAP summary plot showing feature importance."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")


def plot_team_performance(df: pd.DataFrame, output_path: str) -> None:
    """Scatter plot: expected TDs (xScore) vs actual TDs per team."""
    fig, ax = plt.subplots(figsize=(10, 8))
    max_val = max(df["expected_tds"].max(), df["actual_tds"].max()) + 0.5
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="Expected = Actual")
    ax.scatter(df["expected_tds"], df["actual_tds"], s=100, zorder=5)
    for _, row in df.iterrows():
        ax.annotate(row["team"], (row["expected_tds"], row["actual_tds"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_xlabel("Expected TDs (Game xScore)")
    ax.set_ylabel("Actual TDs")
    ax.set_title("Team Performance: xScore vs Reality")
    ax.legend()
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_interactive_heatmap(predictions: pd.DataFrame, output_path: str) -> None:
    """Create interactive Plotly heatmap for web embedding."""
    pivot = predictions.pivot_table(
        values="xscore", index="down", columns="yardline_100", aggfunc="mean"
    )
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=[f"{'1st' if d==1 else '2nd' if d==2 else '3rd' if d==3 else '4th'} Down" for d in pivot.index],
        colorscale="RdYlGn_r",
        zmin=0,
        zmax=1,
        colorbar=dict(title="xScore"),
        hovertemplate="Yard Line: %{x}<br>%{y}<br>xScore: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="xScore: Touchdown Probability by Field Position",
        xaxis_title="Yards from End Zone",
        yaxis_title="Down",
        xaxis=dict(autorange="reversed"),
        width=1200,
        height=400,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")


def plot_coach_aggressiveness(df: pd.DataFrame, output_path: str) -> None:
    """Bar chart of coach 4th-down optimality rate."""
    df_sorted = df.sort_values("pct_optimal", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, len(df_sorted) * 0.4)))
    colors = plt.cm.RdYlGn(df_sorted["pct_optimal"])
    ax.barh(df_sorted["coach"], df_sorted["pct_optimal"], color=colors)
    ax.set_xlabel("% of 4th Downs Following xDecision-Optimal Choice")
    ax.set_title("Coach Aggressiveness: Who Follows the Math?")
    ax.set_xlim(0, 1)
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        ax.text(row["pct_optimal"] + 0.01, i, f"{row['pct_optimal']:.0%}",
                va="center", fontsize=9)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_calibration(metrics: dict, output_path: str) -> None:
    """Plot predicted vs actual calibration curve with Brier/AUC annotations."""
    prob_true = metrics["calibration"]["prob_true"]
    prob_pred = metrics["calibration"]["prob_pred"]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.plot(prob_pred, prob_true, "s-", color="#d62728", label="xScore")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Actual TD Rate")
    ax.set_title("xScore Calibration")
    ax.legend(loc="lower right")
    ax.text(
        0.05, 0.90,
        f"Brier: {metrics['brier_score']:.4f}\nAUC: {metrics['auc_roc']:.4f}",
        transform=ax.transAxes, fontsize=11, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
