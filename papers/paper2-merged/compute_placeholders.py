"""Compute all numbers needed for paper.tex, on the 2018-2025 (256 team-season) sample.

Supersedes papers/paper2-gds/compute_placeholders.py (224-sample, paper2-only numbers)
and the ad hoc computations behind papers/paper3-archetypes/paper3.tex. Run from the
project root with the project venv:

    .venv/Scripts/python papers/paper2-merged/compute_placeholders.py

Requires output/archetype_v2_data.csv (from analyze_archetypes_v2.py) and
output/multinomial_game_gds.csv (from run_multinomial_pipeline.py) to already
be generated for 2018-2025.
"""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

season_df = pd.read_csv("output/archetype_v2_data.csv")
game_df = pd.read_csv("output/multinomial_game_gds.csv")

assert len(season_df) == 256, f"Expected 256 team-seasons, got {len(season_df)}"
season_df = season_df.sort_values(["season", "team"]).reset_index(drop=True)

print("=" * 70)
print("SECTION 1: DESCRIPTIVE STATISTICS (GDS components table)")
print("=" * 70)

season_df["st_val_per_game"] = season_df["st_val"] / season_df["games"]
for col, label in [
    ("off_xvoa_per_game", "Off_xVOA/game"),
    ("def_xvoa_per_game", "Def_xVOA/game"),
    ("st_val_per_game", "ST_Value/game"),
    ("gds_per_game", "Total GDS/game"),
]:
    vals = season_df[col]
    print(f"\n{label}: mean={vals.mean():.3f} SD={vals.std():.3f} "
          f"min={vals.min():.3f} max={vals.max():.3f} skew={stats.skew(vals):.3f}")

shapiro_stat, shapiro_p = stats.shapiro(season_df["gds_per_game"])
print(f"\nGDS/game Shapiro-Wilk: W={shapiro_stat:.4f} p={shapiro_p:.4f}")


print("\n" + "=" * 70)
print("SECTION 2: SEASON-LEVEL CORRELATIONS AND R^2 DECOMPOSITION")
print("=" * 70)

r_gds, p_gds = stats.pearsonr(season_df["gds_per_game"], season_df["win_pct"])
r_off, p_off = stats.pearsonr(season_df["off_xvoa_per_game"], season_df["win_pct"])
r_def, p_def = stats.pearsonr(season_df["def_xvoa_per_game"], season_df["win_pct"])
print(f"GDS/game vs win%:  r={r_gds:.3f} R2={r_gds**2:.1%} p={p_gds:.3e}")
print(f"Off vs win%:       r={r_off:.3f} R2={r_off**2:.1%}")
print(f"Def vs win%:       r={r_def:.3f} R2={r_def**2:.1%}")
print(f"Ratio off:def R^2 = {r_off**2/r_def**2:.1f}:1")
print(f"Synergy (combined R2 - sum of individual R2) = {(r_gds**2 - r_off**2 - r_def**2)*100:.1f}pp")


print("\n" + "=" * 70)
print("SECTION 3: GAME-WINNER PREDICTION ACCURACY")
print("=" * 70)

reg_games = game_df.copy()
game_totals = reg_games.groupby("game_id").agg(n=("posteam", "count")).reset_index()
two_team_games = game_totals[game_totals["n"] == 2]["game_id"]
gg = reg_games[reg_games["game_id"].isin(two_team_games)].copy()

correct = 0
total = 0
for gid, sub in gg.groupby("game_id"):
    if len(sub) != 2:
        continue
    sub = sub.sort_values("gds", ascending=False)
    # winner unknown at game level here without score; skip if not derivable
print("  (Game-winner accuracy requires actual score data joined in; "
      "see run_multinomial_pipeline.py for the authoritative 86.1%-style computation. "
      "Not recomputed here -- cross-check against existing pipeline output before citing.)")


print("\n" + "=" * 70)
print("SECTION 4: OFFENSE-SHARE QUARTILES")
print("=" * 70)

season_df["quartile"] = pd.qcut(season_df["offense_share"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
q_stats = season_df.groupby("quartile", observed=True).agg(
    n=("team", "count"),
    avg_share=("offense_share", "mean"),
    playoff_rate=("made_playoffs", "mean"),
    avg_pw=("playoff_wins", "mean"),
    conf_rate=("playoff_wins", lambda x: (x >= 3).mean()),
    sb_rate=("won_super_bowl", "mean"),
)
print(q_stats.to_string())


print("\n" + "=" * 70)
print("SECTION 5: SUPER BOWL / DEEP PLAYOFF PARTICIPANTS (>=3 playoff wins)")
print("=" * 70)

sb_teams = season_df[season_df["playoff_wins"] >= 3].sort_values(["season", "playoff_wins"], ascending=[True, False])
print(sb_teams[["season", "team", "off_xvoa_per_game", "def_xvoa_per_game", "offense_share",
                "playoff_wins", "won_super_bowl"]].to_string(index=False))

n_sb_winners = season_df["won_super_bowl"].sum()
n_sb_winners_offense_dominant = season_df[(season_df["won_super_bowl"]) & (season_df["offense_share"] > 0)].shape[0]
print(f"\nTotal SB winners: {n_sb_winners}")
print(f"SB winners with positive offense share: {n_sb_winners_offense_dominant}")
print(f"SB winners with negative Def_xVOA/game: "
      f"{season_df[(season_df['won_super_bowl']) & (season_df['def_xvoa_per_game'] < 0)].shape[0]}")


print("\n" + "=" * 70)
print("SECTION 6: HYPOTHESIS TESTS (H1/H2/H3)")
print("=" * 70)

playoff_teams = season_df[season_df["made_playoffs"]]
rho, p_rho = stats.spearmanr(playoff_teams["offense_share"], playoff_teams["playoff_wins"])
print(f"Spearman (playoff teams, n={len(playoff_teams)}): rho={rho:.3f} p={p_rho:.4f}")
boot_rhos = []
rng = np.random.default_rng(42)
arr = playoff_teams[["offense_share", "playoff_wins"]].to_numpy()
for _ in range(5000):
    idx = rng.integers(0, len(arr), len(arr))
    s = arr[idx]
    r, _ = stats.spearmanr(s[:, 0], s[:, 1])
    boot_rhos.append(r)
ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])
print(f"  95% bootstrap CI: [{ci_lo:.3f}, {ci_hi:.3f}]")

off_dom = season_df[season_df["offense_share"] > 0.3]["playoff_wins"]
def_dom = season_df[season_df["offense_share"] < -0.3]["playoff_wins"]
pooled_std = np.sqrt(((len(off_dom)-1)*off_dom.std()**2 + (len(def_dom)-1)*def_dom.std()**2)
                     / (len(off_dom)+len(def_dom)-2))
d = (off_dom.mean() - def_dom.mean()) / pooled_std
print(f"\nCohen's d (all team-seasons): d={d:.3f} n_off={len(off_dom)} (mean={off_dom.mean():.3f}) "
      f"n_def={len(def_dom)} (mean={def_dom.mean():.3f})")
boot_ds = []
off_arr = off_dom.to_numpy()
def_arr = def_dom.to_numpy()
for _ in range(5000):
    o = rng.choice(off_arr, len(off_arr), replace=True)
    de = rng.choice(def_arr, len(def_arr), replace=True)
    ps = np.sqrt(((len(o)-1)*o.std()**2 + (len(de)-1)*de.std()**2) / (len(o)+len(de)-2))
    boot_ds.append((o.mean() - de.mean()) / ps if ps > 0 else 0)
d_lo, d_hi = np.percentile(boot_ds, [2.5, 97.5])
print(f"  95% bootstrap CI: [{d_lo:.3f}, {d_hi:.3f}]")
print(f"  Defense-dominant teams that made playoffs: {season_df[(season_df['offense_share'] < -0.3) & (season_df['made_playoffs'])].shape[0]} / {len(def_dom)}")
print(f"  Max playoff wins among defense-dominant: {def_dom.max()}")

X_lr = sm.add_constant(season_df[["off_xvoa_per_game", "def_xvoa_per_game"]])
y_lr = season_df["won_super_bowl"].astype(int)
logit = sm.Logit(y_lr, X_lr).fit(disp=0)
print(f"\nLogistic: off coef={logit.params['off_xvoa_per_game']:.3f} (p={logit.pvalues['off_xvoa_per_game']:.4f}), "
      f"def coef={logit.params['def_xvoa_per_game']:.3f} (p={logit.pvalues['def_xvoa_per_game']:.4f})")
print(f"  EPV = {int(y_lr.sum())} events / 2 predictors = {y_lr.sum()/2:.1f}")

X_ols = sm.add_constant(season_df[["off_xvoa_per_game", "def_xvoa_per_game"]])
y_ols = season_df["playoff_wins"]
ols = sm.OLS(y_ols, X_ols).fit(cov_type="HC3")
print(f"\nOLS: R2={ols.rsquared:.3f} off coef={ols.params['off_xvoa_per_game']:.3f} (p={ols.pvalues['off_xvoa_per_game']:.4f}) "
      f"def coef={ols.params['def_xvoa_per_game']:.3f} (p={ols.pvalues['def_xvoa_per_game']:.4f})")


print("\n" + "=" * 70)
print("SECTION 7: QUADRANT ANALYSIS (top-quartile GDS, within-elite median split)")
print("=" * 70)

gds_q4_cut = season_df["gds_per_game"].quantile(0.75)
elite = season_df[season_df["gds_per_game"] >= gds_q4_cut].copy()
off_med = elite["off_xvoa_per_game"].median()
def_med = elite["def_xvoa_per_game"].median()
print(f"Elite n={len(elite)}, off median={off_med:.3f}, def median={def_med:.3f}")

def quadrant(row):
    if row["off_xvoa_per_game"] >= off_med and row["def_xvoa_per_game"] >= def_med:
        return "Elite Both"
    elif row["off_xvoa_per_game"] >= off_med:
        return "Offense Only"
    elif row["def_xvoa_per_game"] >= def_med:
        return "Defense Only"
    else:
        return "Neither"

elite["quadrant"] = elite.apply(quadrant, axis=1)
quad_stats = elite.groupby("quadrant").agg(
    n=("team", "count"),
    playoff_rate=("made_playoffs", "mean"),
    avg_pw=("playoff_wins", "mean"),
    conf_rate=("playoff_wins", lambda x: (x >= 3).mean()),
    sb_rate=("won_super_bowl", "mean"),
)
print(quad_stats.to_string())


print("\n" + "=" * 70)
print("SECTION 8: THRESHOLD SENSITIVITY")
print("=" * 70)

for thresh in [0.20, 0.25, 0.30, 0.35, 0.40]:
    d_sub = season_df[season_df["offense_share"] < -thresh]["playoff_wins"]
    print(f"  +/-{thresh:.2f}: n={len(d_sub)} mean_pw={d_sub.mean():.3f}")


print("\n" + "=" * 70)
print("SECTION 9: TIME-WINDOW SENSITIVITY")
print("=" * 70)

windows = [(2018, 2025), (2018, 2023), (2020, 2025), (2021, 2025), (2018, 2021)]
for lo, hi in windows:
    sub = season_df[season_df["season"].between(lo, hi)]
    ro, _ = stats.pearsonr(sub["off_xvoa_per_game"], sub["win_pct"])
    rd, _ = stats.pearsonr(sub["def_xvoa_per_game"], sub["win_pct"])
    print(f"  {lo}-{hi} (n={len(sub)}): ratio={ro**2/rd**2:.1f}:1")


print("\n" + "=" * 70)
print("SECTION 10: SINGLE-TEAM EXCLUSION (KC 2023) AND (SEA 2025)")
print("=" * 70)

for excl_season, excl_team, label in [(2023, "KC", "KC 2023"), (2025, "SEA", "SEA 2025")]:
    sub = season_df[~((season_df["season"] == excl_season) & (season_df["team"] == excl_team))]
    ro, _ = stats.pearsonr(sub["off_xvoa_per_game"], sub["win_pct"])
    rd, _ = stats.pearsonr(sub["def_xvoa_per_game"], sub["win_pct"])
    sub_playoff = sub[sub["made_playoffs"]]
    rho_s, p_s = stats.spearmanr(sub_playoff["offense_share"], sub_playoff["playoff_wins"])
    print(f"  Excluding {label}: ratio={ro**2/rd**2:.1f}:1, Spearman rho={rho_s:.3f} (p={p_s:.4f})")


print("\n" + "=" * 70)
print("SECTION 11: ERA COMPARISON (2018-2020 vs 2021-2025)")
print("=" * 70)

early = season_df[season_df["season"].between(2018, 2020)]
late = season_df[season_df["season"].between(2021, 2025)]
early_playoff = early[early["made_playoffs"]]
late_playoff = late[late["made_playoffs"]]
rho_e, p_e = stats.spearmanr(early_playoff["offense_share"], early_playoff["playoff_wins"])
rho_l, p_l = stats.spearmanr(late_playoff["offense_share"], late_playoff["playoff_wins"])
r_off_e, _ = stats.pearsonr(early["off_xvoa_per_game"], early["win_pct"])
r_def_e, _ = stats.pearsonr(early["def_xvoa_per_game"], early["win_pct"])
r_off_l, _ = stats.pearsonr(late["off_xvoa_per_game"], late["win_pct"])
r_def_l, _ = stats.pearsonr(late["def_xvoa_per_game"], late["win_pct"])
print(f"  2018-2020 (n_playoff={len(early_playoff)}): Spearman rho={rho_e:.3f} (p={p_e:.4f}); "
      f"Pearson off r={r_off_e:.3f} def r={r_def_e:.3f}")
print(f"  2021-2025 (n_playoff={len(late_playoff)}): Spearman rho={rho_l:.3f} (p={p_l:.4f}); "
      f"Pearson off r={r_off_l:.3f} def r={r_def_l:.3f}")


print("\n" + "=" * 70)
print("SECTION 12: PER-SEASON CORRELATION STABILITY")
print("=" * 70)

per_season_r = {}
for yr in range(2018, 2026):
    sub = season_df[season_df["season"] == yr]
    r, p = stats.pearsonr(sub["gds_per_game"], sub["win_pct"])
    per_season_r[yr] = r
    print(f"  {yr}: r={r:.3f} R2={r**2:.1%}")
r_vals = list(per_season_r.values())
print(f"  Range: {min(r_vals):.3f} to {max(r_vals):.3f}")


print("\n" + "=" * 70)
print("SECTION 13: SPLIT-HALF RELIABILITY")
print("=" * 70)

split_rhos = []
for yr in range(2018, 2026):
    yr_games = game_df[game_df["season"] == yr].copy()
    yr_games["week"] = yr_games["game_id"].str.split("_").str[1].astype(int)
    odd_gds = yr_games[yr_games["week"] % 2 == 1].groupby("posteam")["gds"].mean().rename("odd")
    even_gds = yr_games[yr_games["week"] % 2 == 0].groupby("posteam")["gds"].mean().rename("even")
    combined = pd.concat([odd_gds, even_gds], axis=1).dropna()
    if len(combined) >= 10:
        rho, _ = stats.spearmanr(combined["odd"], combined["even"])
        split_rhos.append(rho)
        print(f"  {yr}: rho={rho:.3f}")
mean_split = float(np.mean(split_rhos))
print(f"  Mean split-half rho: {mean_split:.3f}")
print(f"  Spearman-Brown prophecy: {2*mean_split/(1+mean_split):.3f}")

print("\n" + "=" * 70)
print("SECTION 14: YEAR-OVER-YEAR RANK STABILITY")
print("=" * 70)

yoy_rhos = []
for yr in range(2018, 2025):
    yr1 = season_df[season_df["season"] == yr][["team", "gds_per_game"]].set_index("team")
    yr2 = season_df[season_df["season"] == yr + 1][["team", "gds_per_game"]].set_index("team")
    common = yr1.join(yr2, lsuffix="_y1", rsuffix="_y2").dropna()
    rho, _ = stats.spearmanr(common["gds_per_game_y1"], common["gds_per_game_y2"])
    yoy_rhos.append(rho)
    print(f"  {yr}->{yr+1}: rho={rho:.3f}")
print(f"  Mean rho: {np.mean(yoy_rhos):.3f}  Range: {min(yoy_rhos):.3f} to {max(yoy_rhos):.3f}")


print("\n" + "=" * 70)
print("SECTION 15: LUCK ANALYSIS (2025 season, using full-sample GDS-win% regression)")
print("=" * 70)

slope, intercept, r_fit, p_fit, se_fit = stats.linregress(season_df["gds_per_game"], season_df["win_pct"])
s2025 = season_df[season_df["season"] == 2025].copy()
s2025["implied_win_pct"] = intercept + slope * s2025["gds_per_game"]
s2025["implied_wins"] = s2025["implied_win_pct"] * s2025["reg_games"]
s2025["actual_wins"] = s2025["reg_wins"]
s2025["divergence"] = s2025["actual_wins"] - s2025["implied_wins"]
s2025_sorted = s2025.sort_values("divergence", ascending=False)
print("Luckiest (2025):")
for _, row in s2025_sorted.head(3).iterrows():
    print(f"  {row['team']}: actual={row['actual_wins']:.0f} implied={row['implied_wins']:.1f} div={row['divergence']:+.1f}")
print("Unluckiest (2025):")
for _, row in s2025_sorted.tail(3).iterrows():
    print(f"  {row['team']}: actual={row['actual_wins']:.0f} implied={row['implied_wins']:.1f} div={row['divergence']:+.1f}")


print("\n" + "=" * 70)
print("SECTION 16: POINT-DIFFERENTIAL BASELINE COMPARISON")
print("=" * 70)
try:
    import nfl_data_py as nfl
    schedules = nfl.import_schedules(list(range(2018, 2026)))
    reg = schedules[schedules["game_type"] == "REG"].copy()
    records = []
    for _, g in reg.iterrows():
        records.append({"season": g["season"], "team": g["home_team"], "pf": g["home_score"], "pa": g["away_score"]})
        records.append({"season": g["season"], "team": g["away_team"], "pf": g["away_score"], "pa": g["home_score"]})
    pts_df = pd.DataFrame(records)
    pts_season = pts_df.groupby(["season", "team"]).agg(
        total_pf=("pf", "sum"), total_pa=("pa", "sum"), games=("pf", "count")
    ).reset_index()
    pts_season["pt_diff_per_game"] = (pts_season["total_pf"] - pts_season["total_pa"]) / pts_season["games"]
    merged_pd = season_df.merge(pts_season[["season", "team", "pt_diff_per_game"]], on=["season", "team"], how="left")
    valid = merged_pd.dropna(subset=["pt_diff_per_game"])
    r_ptdiff, _ = stats.pearsonr(valid["pt_diff_per_game"], valid["win_pct"])
    print(f"Point differential/game vs win%: r={r_ptdiff:.3f} R2={r_ptdiff**2:.1%}")
    print(f"GDS/game vs win%:               r={r_gds:.3f} R2={r_gds**2:.1%}")
    print(f"Gap: {(r_ptdiff**2 - r_gds**2)*100:.1f}pp")
except Exception as e:
    print("  ERROR:", e)

print("\nDONE")
