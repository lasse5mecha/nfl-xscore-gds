"""Generate all thesis figures from saved model and output data."""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import xgboost as xgb
import shap
import joblib
from pathlib import Path
from sklearn.calibration import calibration_curve
from src.data import FEATURE_COLUMNS, OUTCOME_CLASSES, OUTCOME_TO_INDEX
from src.model import predict_multinomial

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
})

FIGDIR = Path("thesis/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

# Load data — use the authoritative archetype CSV (has corrected SB winners)
print("Loading data...")
gds_data = pd.read_csv("output/archetype_v2_data.csv")
game_gds = pd.read_csv("output/multinomial_game_gds.csv")

# Load model for SHAP and calibration plots
print("Loading model and calibrators...")
model = xgb.XGBClassifier()
model.load_model("models/xscore_multinomial.json")
calibrators = joblib.load("models/xscore_multinomial_calibrators.pkl")


# ============================================================
# FIGURE 1: Calibration Reliability Diagram (4-panel)
# ============================================================
def fig1_calibration():
    print("  [1/10] Calibration reliability diagram...")
    from src.data import load_play_by_play, filter_offensive_snaps, add_drive_outcome_target, engineer_features, split_by_season

    raw = load_play_by_play([2025])
    df = filter_offensive_snaps(raw)
    df = add_drive_outcome_target(df)
    df = engineer_features(df)
    df = df.dropna(subset=FEATURE_COLUMNS + ["drive_outcome"])

    X_test = df[FEATURE_COLUMNS]
    y_test = df["drive_outcome"].values

    raw_probs = model.predict_proba(X_test)
    cal_probs = predict_multinomial(model, X_test, calibrators=calibrators)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    class_names = ["Punt/Other", "Touchdown", "Field Goal", "Turnover"]

    for idx, (ax, name) in enumerate(zip(axes.flat, class_names)):
        y_binary = (y_test == idx).astype(int)

        # Raw (uncalibrated)
        prob_true_raw, prob_pred_raw = calibration_curve(
            y_binary, raw_probs[:, idx], n_bins=10, strategy="quantile"
        )
        # Calibrated
        prob_true_cal, prob_pred_cal = calibration_curve(
            y_binary, cal_probs[:, idx], n_bins=10, strategy="quantile"
        )

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1)
        ax.plot(prob_pred_raw, prob_true_raw, "s--", color="#d62728",
                markersize=5, label="Pre-calibration", alpha=0.7)
        ax.plot(prob_pred_cal, prob_true_cal, "o-", color="#1f77b4",
                markersize=6, label="Post-calibration")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(f"{name} (base rate: {y_binary.mean():.1%})")
        ax.legend(loc="lower right", framealpha=0.9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Per-Class Calibration Reliability Diagrams (2025 Test Set)", y=1.01)
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig1_calibration.pdf")
    plt.savefig(FIGDIR / "fig1_calibration.png")
    plt.close()
    return X_test


# ============================================================
# FIGURE 2: xScore Field Heatmap
# ============================================================
def fig2_field_heatmap():
    print("  [2/10] xScore field heatmap...")
    grid = []
    for down in [1, 2, 3, 4]:
        for yd in range(1, 100):
            grid.append({
                "down": down, "ydstogo": 10, "yardline_100": yd,
                "score_diff": 0, "half_seconds_remaining": 900,
                "goal_to_go": 1 if yd <= 10 else 0,
                "red_zone": 1 if yd <= 20 else 0,
            })
    grid_df = pd.DataFrame(grid)
    probs = predict_multinomial(model, grid_df[FEATURE_COLUMNS], calibrators=calibrators)
    grid_df["p_td"] = probs[:, OUTCOME_TO_INDEX["td"]]

    pivot = grid_df.pivot_table(values="p_td", index="down", columns="yardline_100")

    fig, ax = plt.subplots(figsize=(14, 3.5))
    im = ax.imshow(
        pivot.values, aspect="auto", cmap="RdYlGn_r",
        vmin=0, vmax=0.85, extent=[1, 99, 4.5, 0.5],
        interpolation="bilinear"
    )
    ax.set_xlabel("Yards from opponent's end zone")
    ax.set_ylabel("Down")
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(["1st", "2nd", "3rd", "4th"])
    cbar = plt.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("P(Touchdown)")
    ax.set_title("xScore: Touchdown Probability by Field Position and Down")
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig2_field_heatmap.pdf")
    plt.savefig(FIGDIR / "fig2_field_heatmap.png")
    plt.close()


# ============================================================
# FIGURE 3: Off/Def xVOA vs Win% (side-by-side)
# ============================================================
def fig3_xvoa_vs_winpct():
    print("  [3/10] xVOA vs Win% scatter...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Offense
    ax1.scatter(gds_data["off_xvoa_per_game"], gds_data["win_pct"],
                alpha=0.5, s=30, color="#1f77b4", edgecolors="none")
    z = np.polyfit(gds_data["off_xvoa_per_game"], gds_data["win_pct"], 1)
    x_range = np.linspace(gds_data["off_xvoa_per_game"].min(),
                          gds_data["off_xvoa_per_game"].max(), 100)
    ax1.plot(x_range, np.polyval(z, x_range), "r-", linewidth=2, alpha=0.8)
    r_off = round(gds_data["off_xvoa_per_game"].corr(gds_data["win_pct"]), 3)
    ax1.set_xlabel("Offensive xVOA / game")
    ax1.set_ylabel("Win percentage")
    ax1.set_title(f"Offense vs Win%\nr = {r_off:.3f}, R² = {r_off**2:.1%}")
    ax1.grid(True, alpha=0.3)

    # Defense
    ax2.scatter(gds_data["def_xvoa_per_game"], gds_data["win_pct"],
                alpha=0.5, s=30, color="#ff7f0e", edgecolors="none")
    z = np.polyfit(gds_data["def_xvoa_per_game"], gds_data["win_pct"], 1)
    x_range = np.linspace(gds_data["def_xvoa_per_game"].min(),
                          gds_data["def_xvoa_per_game"].max(), 100)
    ax2.plot(x_range, np.polyval(z, x_range), "r-", linewidth=2, alpha=0.8)
    r_def = round(gds_data["def_xvoa_per_game"].corr(gds_data["win_pct"]), 3)
    ax2.set_xlabel("Defensive xVOA / game")
    ax2.set_ylabel("Win percentage")
    ax2.set_title(f"Defense vs Win%\nr = {r_def:.3f}, R² = {r_def**2:.1%}")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Offensive vs Defensive Contribution to Win Percentage (N = 224)", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(FIGDIR / "fig3_xvoa_vs_winpct.pdf")
    plt.savefig(FIGDIR / "fig3_xvoa_vs_winpct.png")
    plt.close()


# ============================================================
# FIGURE 4: Quartile Bar Chart
# ============================================================
def fig4_quartile_bars():
    print("  [4/10] Quartile bar chart...")
    playoff_teams = gds_data[gds_data["made_playoffs"]].copy()

    gds_data_sorted = gds_data.sort_values("offense_share")
    n = len(gds_data_sorted)
    q_size = n // 4
    gds_data_sorted["quartile"] = pd.cut(
        gds_data_sorted["offense_share"],
        bins=[-np.inf,
              gds_data_sorted["offense_share"].quantile(0.25),
              gds_data_sorted["offense_share"].quantile(0.50),
              gds_data_sorted["offense_share"].quantile(0.75),
              np.inf],
        labels=["Q1\n(Defense)", "Q2", "Q3", "Q4\n(Offense)"]
    )

    metrics = gds_data_sorted.groupby("quartile", observed=True).agg(
        playoff_pct=("made_playoffs", "mean"),
        avg_pw=("playoff_wins", "mean"),
        sb_pct=("won_super_bowl", "mean"),
    ).reset_index()

    x = np.arange(4)
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars1 = ax.bar(x - width, metrics["playoff_pct"] * 100, width,
                   label="Playoff %", color="#1f77b4", alpha=0.85)
    bars2 = ax.bar(x, metrics["avg_pw"] / metrics["avg_pw"].max() * 100, width,
                   label="Avg PW (scaled)", color="#2ca02c", alpha=0.85)
    bars3 = ax.bar(x + width, metrics["sb_pct"] * 100, width,
                   label="SB Win %", color="#d62728", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics["quartile"])
    ax.set_ylabel("Percentage / Scaled value")
    ax.set_title("Postseason Outcomes by Offense-Share Quartile")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate key values
    for i, row in metrics.iterrows():
        ax.text(x[i] - width, row["playoff_pct"] * 100 + 1,
                f'{row["playoff_pct"]*100:.0f}%', ha="center", fontsize=9)
        ax.text(x[i], row["avg_pw"] / metrics["avg_pw"].max() * 100 + 1,
                f'{row["avg_pw"]:.2f}', ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGDIR / "fig4_quartile_bars.pdf")
    plt.savefig(FIGDIR / "fig4_quartile_bars.png")
    plt.close()


# ============================================================
# FIGURE 5: SHAP Beeswarm Plot
# ============================================================
def fig5_shap(X_sample):
    print("  [5/10] SHAP beeswarm plot...")
    # Use a subsample for speed
    if len(X_sample) > 5000:
        X_sub = X_sample.sample(5000, random_state=42)
    else:
        X_sub = X_sample

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sub)

    # shap_values is list of 4 arrays (one per class); use TD class
    td_idx = OUTCOME_TO_INDEX["td"]
    td_shap = shap_values[td_idx] if isinstance(shap_values, list) else shap_values[:, :, td_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    shap.summary_plot(td_shap, X_sub, show=False, plot_size=None)
    plt.title("SHAP Feature Importance (Touchdown Class)")
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig5_shap_beeswarm.pdf")
    plt.savefig(FIGDIR / "fig5_shap_beeswarm.png")
    plt.close("all")


# ============================================================
# FIGURE 6: GDS/game vs Win%
# ============================================================
def fig6_gds_vs_winpct():
    print("  [6/10] GDS/game vs Win%...")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(gds_data["gds_per_game"], gds_data["win_pct"],
               alpha=0.5, s=30, color="#2ca02c", edgecolors="none")
    z = np.polyfit(gds_data["gds_per_game"], gds_data["win_pct"], 1)
    x_range = np.linspace(gds_data["gds_per_game"].min(),
                          gds_data["gds_per_game"].max(), 100)
    ax.plot(x_range, np.polyval(z, x_range), "r-", linewidth=2)
    r = gds_data["gds_per_game"].corr(gds_data["win_pct"])
    ax.set_xlabel("GDS / game")
    ax.set_ylabel("Win percentage")
    ax.set_title(f"GDS/game vs Win Percentage\nr = {r:.3f}, R² = {r**2:.1%}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig6_gds_vs_winpct.pdf")
    plt.savefig(FIGDIR / "fig6_gds_vs_winpct.png")
    plt.close()


# ============================================================
# FIGURE 7: Offense Share vs Playoff Wins (with SB winners annotated)
# ============================================================
def fig7_offense_share_vs_pw():
    print("  [7/10] Offense share vs playoff wins...")
    playoff = gds_data[gds_data["made_playoffs"]].copy()

    fig, ax = plt.subplots(figsize=(9, 6))

    # Non-SB winners
    non_sb = playoff[~playoff["won_super_bowl"]]
    sb = playoff[playoff["won_super_bowl"]]

    ax.scatter(non_sb["offense_share"], non_sb["playoff_wins"],
               alpha=0.4, s=40, color="#1f77b4", label="Playoff teams")
    ax.scatter(sb["offense_share"], sb["playoff_wins"],
               s=120, color="#d62728", marker="*", zorder=5,
               label="Super Bowl winners", edgecolors="black", linewidths=0.5)

    # Annotate SB winners with manual offsets to avoid overlap
    offsets = {
        ("NE", 2018): (-30, 12),
        ("KC", 2019): (-30, -14),
        ("TB", 2020): (-35, 10),
        ("LA", 2021): (6, -14),
        ("KC", 2022): (-35, 10),
        ("KC", 2023): (-35, 10),
        ("PHI", 2024): (-30, 10),
    }
    for _, row in sb.iterrows():
        label = f"{row['team']} '{int(row['season']) % 100}"
        key = (row['team'], int(row['season']))
        xytext = offsets.get(key, (6, 4))
        ax.annotate(label, (row["offense_share"], row["playoff_wins"]),
                    textcoords="offset points", xytext=xytext, fontsize=8,
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=-0.3, color="red", linestyle=":", alpha=0.6, label="Def-dominant threshold")
    ax.set_xlabel("Offense share")
    ax.set_ylabel("Playoff wins")
    ax.set_title("Offense Share vs Playoff Wins (Playoff Teams, 2018–2024)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig7_offense_share_pw.pdf")
    plt.savefig(FIGDIR / "fig7_offense_share_pw.png")
    plt.close()


# ============================================================
# FIGURE 8: Era Comparison
# ============================================================
def fig8_era_comparison():
    print("  [8/10] Era comparison...")
    early = gds_data[gds_data["season"].between(2018, 2020)].copy()
    late = gds_data[gds_data["season"].between(2021, 2024)].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Offense correlation by era
    for era_df, label, color in [(early, "2018–2020", "#1f77b4"), (late, "2021–2024", "#ff7f0e")]:
        r = era_df["off_xvoa_per_game"].corr(era_df["win_pct"])
        ax1.scatter(era_df["off_xvoa_per_game"], era_df["win_pct"],
                    alpha=0.4, s=25, color=color, label=f"{label} (r={r:.3f})")
        z = np.polyfit(era_df["off_xvoa_per_game"], era_df["win_pct"], 1)
        x_r = np.linspace(era_df["off_xvoa_per_game"].min(),
                          era_df["off_xvoa_per_game"].max(), 50)
        ax1.plot(x_r, np.polyval(z, x_r), color=color, linewidth=2, alpha=0.8)

    ax1.set_xlabel("Offensive xVOA / game")
    ax1.set_ylabel("Win percentage")
    ax1.set_title("Offensive Predictive Power by Era")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Defense correlation by era
    for era_df, label, color in [(early, "2018–2020", "#1f77b4"), (late, "2021–2024", "#ff7f0e")]:
        r = era_df["def_xvoa_per_game"].corr(era_df["win_pct"])
        ax2.scatter(era_df["def_xvoa_per_game"], era_df["win_pct"],
                    alpha=0.4, s=25, color=color, label=f"{label} (r={r:.3f})")
        z = np.polyfit(era_df["def_xvoa_per_game"], era_df["win_pct"], 1)
        x_r = np.linspace(era_df["def_xvoa_per_game"].min(),
                          era_df["def_xvoa_per_game"].max(), 50)
        ax2.plot(x_r, np.polyval(z, x_r), color=color, linewidth=2, alpha=0.8)

    ax2.set_xlabel("Defensive xVOA / game")
    ax2.set_ylabel("Win percentage")
    ax2.set_title("Defensive Predictive Power by Era")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Era Comparison: Correlation Shifts (2018–2020 vs 2021–2024)", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(FIGDIR / "fig8_era_comparison.pdf")
    plt.savefig(FIGDIR / "fig8_era_comparison.png")
    plt.close()


# ============================================================
# FIGURE 9: GDS Pipeline Flowchart
# ============================================================
def fig9_pipeline_flowchart():
    print("  [9/10] GDS pipeline flowchart...")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (1, 2, "Play-Level\nGame State"),
        (3, 2, "xScore\n(XGBoost)"),
        (5, 2, "4-Class\nProbabilities"),
        (7, 2, "xEP\n(Expected Pts)"),
        (9, 2, "Drive xVOA\n(Actual − xEP)"),
        (11, 2, "Season GDS\n(Off + Def + ST)"),
    ]

    for x, y, text in boxes:
        bbox = mpatches.FancyBboxPatch(
            (x - 0.7, y - 0.55), 1.4, 1.1,
            boxstyle="round,pad=0.1",
            facecolor="#e8f4fd", edgecolor="#1f77b4", linewidth=1.5
        )
        ax.add_patch(bbox)
        ax.text(x, y, text, ha="center", va="center", fontsize=9, fontweight="bold")

    # Arrows
    for i in range(len(boxes) - 1):
        ax.annotate("", xy=(boxes[i+1][0] - 0.75, 2),
                    xytext=(boxes[i][0] + 0.75, 2),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#333"))

    # Labels below
    sublabels = [
        (1, 0.9, "down, ydstogo,\nyardline, score_diff, ..."),
        (3, 0.9, "Multinomial\n(multi:softprob)"),
        (5, 0.9, "P(TD), P(FG),\nP(TO), P(Punt)"),
        (7, 0.9, "7·P(TD) + 3·P(FG)\n− penalty·P(TO)"),
        (9, 0.9, "pts_actual − xEP\nper drive"),
        (11, 0.9, "Σ per team\nover 17 games"),
    ]
    for x, y, text in sublabels:
        ax.text(x, y, text, ha="center", va="center", fontsize=7.5,
                color="#555", style="italic")

    ax.set_title("GDS Computation Pipeline", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig9_pipeline_flowchart.pdf")
    plt.savefig(FIGDIR / "fig9_pipeline_flowchart.png")
    plt.close()


# ============================================================
# FIGURE 10: Archetype Quadrant Diagram
# ============================================================
def fig10_quadrant_diagram():
    print("  [10/10] Archetype quadrant diagram...")
    # Top-quartile teams
    threshold = gds_data["gds_per_game"].quantile(0.75)
    elite = gds_data[gds_data["gds_per_game"] >= threshold].copy()

    off_med = elite["off_xvoa_per_game"].median()
    def_med = elite["def_xvoa_per_game"].median()

    fig, ax = plt.subplots(figsize=(8, 7))

    # Color by quadrant
    colors = []
    labels = []
    for _, row in elite.iterrows():
        if row["off_xvoa_per_game"] >= off_med and row["def_xvoa_per_game"] >= def_med:
            colors.append("#2ca02c")
            labels.append("Elite Both")
        elif row["off_xvoa_per_game"] >= off_med:
            colors.append("#1f77b4")
            labels.append("Offense Only")
        elif row["def_xvoa_per_game"] >= def_med:
            colors.append("#ff7f0e")
            labels.append("Defense Only")
        else:
            colors.append("#9467bd")
            labels.append("Neither")

    # Size by playoff wins
    sizes = 50 + elite["playoff_wins"] * 60

    ax.scatter(elite["off_xvoa_per_game"], elite["def_xvoa_per_game"],
               c=colors, s=sizes, alpha=0.7, edgecolors="black", linewidths=0.5)

    # Median lines
    ax.axvline(off_med, color="gray", linestyle="--", alpha=0.7)
    ax.axhline(def_med, color="gray", linestyle="--", alpha=0.7)

    # Quadrant labels
    ax.text(elite["off_xvoa_per_game"].max() * 0.85, elite["def_xvoa_per_game"].max() * 0.9,
            "ELITE BOTH", fontsize=10, fontweight="bold", color="#2ca02c", alpha=0.7)
    ax.text(elite["off_xvoa_per_game"].max() * 0.85, elite["def_xvoa_per_game"].min() * 0.7,
            "OFFENSE\nONLY", fontsize=10, fontweight="bold", color="#1f77b4", alpha=0.7)
    ax.text(elite["off_xvoa_per_game"].min() * 0.7, elite["def_xvoa_per_game"].max() * 0.9,
            "DEFENSE\nONLY", fontsize=10, fontweight="bold", color="#ff7f0e", alpha=0.7)
    ax.text(elite["off_xvoa_per_game"].min() * 0.7, elite["def_xvoa_per_game"].min() * 0.7,
            "NEITHER", fontsize=10, fontweight="bold", color="#9467bd", alpha=0.7)

    # Annotate SB winners
    sb_elite = elite[elite["won_super_bowl"]]
    for _, row in sb_elite.iterrows():
        ax.annotate(f"{row['team']} '{int(row['season'])%100}",
                    (row["off_xvoa_per_game"], row["def_xvoa_per_game"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8,
                    fontweight="bold", color="#d62728")

    ax.set_xlabel("Offensive xVOA / game")
    ax.set_ylabel("Defensive xVOA / game")
    ax.set_title("Elite Team Quadrant Analysis (Top-Quartile GDS)")
    ax.grid(True, alpha=0.3)

    # Legend
    legend_elements = [
        mpatches.Patch(color="#2ca02c", alpha=0.7, label="Elite Both"),
        mpatches.Patch(color="#1f77b4", alpha=0.7, label="Offense Only"),
        mpatches.Patch(color="#ff7f0e", alpha=0.7, label="Defense Only"),
        mpatches.Patch(color="#9467bd", alpha=0.7, label="Neither"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig10_quadrant_diagram.pdf")
    plt.savefig(FIGDIR / "fig10_quadrant_diagram.png")
    plt.close()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\nGenerating thesis figures...")
    print("=" * 50)

    X_test = fig1_calibration()
    fig2_field_heatmap()
    fig3_xvoa_vs_winpct()
    fig4_quartile_bars()
    fig5_shap(X_test)
    fig6_gds_vs_winpct()
    fig7_offense_share_vs_pw()
    fig8_era_comparison()
    fig9_pipeline_flowchart()
    fig10_quadrant_diagram()

    print("=" * 50)
    print(f"All figures saved to {FIGDIR}/")
    print("Files: fig1_calibration, fig2_field_heatmap, fig3_xvoa_vs_winpct,")
    print("       fig4_quartile_bars, fig5_shap_beeswarm, fig6_gds_vs_winpct,")
    print("       fig7_offense_share_pw, fig8_era_comparison,")
    print("       fig9_pipeline_flowchart, fig10_quadrant_diagram")
    print("  (each in .pdf and .png)")
