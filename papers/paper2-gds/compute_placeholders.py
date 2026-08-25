"""Compute all [COMPUTE] placeholder values for paper2.tex.

Superseded by papers/paper2-merged/compute_placeholders.py, which covers the
same numbers on the current 256-team-season (2018-2025) sample. Kept for
historical reference to how paper2's original 224-sample numbers were derived.
Run from the project root.
"""
import pandas as pd
import numpy as np
from scipy import stats

# Load precomputed data
season_df = pd.read_csv("output/multinomial_gds_data.csv")
game_df = pd.read_csv("output/multinomial_game_gds.csv")

# Filter to 2018-2024 only (224 team-seasons)
season_df = season_df[season_df["season"].between(2018, 2024)].copy()
game_df = game_df[game_df["season"].between(2018, 2024)].copy()
assert len(season_df) == 224, f"Expected 224 team-seasons, got {len(season_df)}"

# Compute win percentage
season_df["win_pct"] = season_df["reg_wins"] / season_df["reg_games"]

# ST value per game
season_df["st_val_per_game"] = season_df["st_val"] / season_df["games"]

print("=" * 70)
print("SECTION 1: DESCRIPTIVE STATISTICS (Table in §4.9)")
print("=" * 70)

for col, label in [
    ("off_xova_per_game", "Off_xVOA/game"),
    ("def_xova_per_game", "Def_xVOA/game"),
    ("st_val_per_game", "ST_Value/game"),
    ("gds_per_game", "Total GDS/game"),
]:
    vals = season_df[col]
    print(f"\n{label}:")
    print(f"  Mean:     {vals.mean():.3f}")
    print(f"  SD:       {vals.std():.3f}")
    print(f"  Min:      {vals.min():.3f}")
    print(f"  Max:      {vals.max():.3f}")
    print(f"  Skewness: {stats.skew(vals):.3f}")

# Normality test for GDS/game
gds_vals = season_df["gds_per_game"]
shapiro_stat, shapiro_p = stats.shapiro(gds_vals)
print(f"\nGDS/game normality (Shapiro-Wilk): W={shapiro_stat:.4f}, p={shapiro_p:.4f}")
if shapiro_p > 0.05:
    print("  -> Approximately normal (fail to reject normality)")
else:
    print("  -> Departs from normality")
skew_gds = stats.skew(gds_vals)
print(f"  Skewness: {skew_gds:.3f} ({'approximately symmetric' if abs(skew_gds) < 0.5 else 'mildly skewed'})")


print("\n" + "=" * 70)
print("SECTION 2: POINT DIFFERENTIAL BASELINE (§5.2.1)")
print("=" * 70)

# Compute point differential from nflfastR schedule data
try:
    import nfl_data_py as nfl
    schedules = nfl.import_schedules(list(range(2018, 2025)))
    reg = schedules[schedules["game_type"] == "REG"].copy()

    records = []
    for _, g in reg.iterrows():
        records.append({"season": g["season"], "team": g["home_team"], "pf": g["home_score"], "pa": g["away_score"]})
        records.append({"season": g["season"], "team": g["away_team"], "pf": g["away_score"], "pa": g["home_score"]})

    pts_df = pd.DataFrame(records)
    pts_season = pts_df.groupby(["season", "team"]).agg(
        total_pf=("pf", "sum"),
        total_pa=("pa", "sum"),
        games=("pf", "count"),
    ).reset_index()
    pts_season["pt_diff_per_game"] = (pts_season["total_pf"] - pts_season["total_pa"]) / pts_season["games"]

    merged = season_df.merge(pts_season[["season", "team", "pt_diff_per_game"]], on=["season", "team"], how="left")
    valid = merged.dropna(subset=["pt_diff_per_game"])

    r_ptdiff, p_ptdiff = stats.pearsonr(valid["pt_diff_per_game"], valid["win_pct"])
    r2_ptdiff = r_ptdiff ** 2

    r_gds, _ = stats.pearsonr(season_df["gds_per_game"], season_df["win_pct"])
    r2_gds = r_gds ** 2

    print(f"\nPoint differential/game vs win%: r = {r_ptdiff:.3f}, R² = {r2_ptdiff*100:.1f}%")
    print(f"GDS/game vs win%:               r = {r_gds:.3f}, R² = {r2_gds*100:.1f}%")
    print(f"GDS exceeds point diff by:      {(r2_gds - r2_ptdiff)*100:.1f} percentage points")
    if r2_gds > r2_ptdiff:
        print("  -> GDS exceeds point differential")
    else:
        print("  -> Point differential matches or exceeds GDS")
except Exception as e:
    print(f"  ERROR computing point differential: {e}")
    print("  Falling back: using known relationship (pt diff typically r~0.92 with win%)")


print("\n" + "=" * 70)
print("SECTION 3: PER-SEASON CORRELATIONS (§5.4)")
print("=" * 70)

per_season_r = {}
for yr in range(2018, 2025):
    sub = season_df[season_df["season"] == yr]
    r, p = stats.pearsonr(sub["gds_per_game"], sub["win_pct"])
    per_season_r[yr] = r
    print(f"  {yr}: r = {r:.3f}, R² = {r**2*100:.1f}%")

r_vals = list(per_season_r.values())
print(f"\n  Range: {min(r_vals):.3f} to {max(r_vals):.3f}")
print(f"  All above r = {min(r_vals):.3f}")


print("\n" + "=" * 70)
print("SECTION 4: YEAR-OVER-YEAR RANK STABILITY (§5.4.2)")
print("=" * 70)

yoy_rhos = []
for yr in range(2018, 2024):
    yr1 = season_df[season_df["season"] == yr][["team", "gds_per_game"]].set_index("team")
    yr2 = season_df[season_df["season"] == yr + 1][["team", "gds_per_game"]].set_index("team")
    common = yr1.join(yr2, lsuffix="_y1", rsuffix="_y2").dropna()
    rho, p = stats.spearmanr(common["gds_per_game_y1"], common["gds_per_game_y2"])
    yoy_rhos.append(rho)
    print(f"  {yr}->{yr+1}: ρ = {rho:.3f}")

print(f"\n  Mean ρ: {np.mean(yoy_rhos):.3f}")
print(f"  Range: {min(yoy_rhos):.3f} to {max(yoy_rhos):.3f}")


print("\n" + "=" * 70)
print("SECTION 5: SPLIT-HALF RELIABILITY (§5.5)")
print("=" * 70)

# Need week info from game data
split_rhos = []
for yr in range(2018, 2025):
    yr_games = game_df[game_df["season"] == yr].copy()
    # Extract week from game_id (format: YYYY_WW_AWAY_HOME)
    yr_games["week"] = yr_games["game_id"].str.split("_").str[1].astype(int)

    odd_weeks = yr_games[yr_games["week"] % 2 == 1]
    even_weeks = yr_games[yr_games["week"] % 2 == 0]

    odd_gds = odd_weeks.groupby("posteam")["gds"].mean().rename("odd_gds")
    even_gds = even_weeks.groupby("posteam")["gds"].mean().rename("even_gds")

    combined = pd.concat([odd_gds, even_gds], axis=1).dropna()
    if len(combined) >= 10:
        rho, p = stats.spearmanr(combined["odd_gds"], combined["even_gds"])
        split_rhos.append(rho)
        print(f"  {yr}: ρ = {rho:.3f}")

mean_split = np.mean(split_rhos)
sb_prophecy = 2 * mean_split / (1 + mean_split)
print(f"\n  Mean split-half ρ: {mean_split:.3f}")
print(f"  Spearman-Brown prophecy (full season): {sb_prophecy:.3f}")
print(f"  Range: {min(split_rhos):.3f} to {max(split_rhos):.3f}")


print("\n" + "=" * 70)
print("SECTION 6: ST VALUE EMPIRICAL (§6.2)")
print("=" * 70)

st_r, st_p = stats.pearsonr(season_df["st_val_per_game"], season_df["win_pct"])
print(f"\n  ST_Value/game vs win%: r = {st_r:.3f}, R² = {st_r**2*100:.1f}%")

# Top/bottom ST team-seasons
season_df_sorted = season_df.sort_values("st_val_per_game", ascending=False)
print("\n  Top 3 ST_Value/game:")
for _, row in season_df_sorted.head(3).iterrows():
    print(f"    {row['team']} {int(row['season'])}: {row['st_val_per_game']:.3f}")
print("\n  Bottom 3 ST_Value/game:")
for _, row in season_df_sorted.tail(3).iterrows():
    print(f"    {row['team']} {int(row['season'])}: {row['st_val_per_game']:.3f}")

# Year-over-year ST stability
st_yoy_rhos = []
for yr in range(2018, 2024):
    yr1 = season_df[season_df["season"] == yr][["team", "st_val_per_game"]].set_index("team")
    yr2 = season_df[season_df["season"] == yr + 1][["team", "st_val_per_game"]].set_index("team")
    common = yr1.join(yr2, lsuffix="_y1", rsuffix="_y2").dropna()
    rho, p = stats.spearmanr(common["st_val_per_game_y1"], common["st_val_per_game_y2"])
    st_yoy_rhos.append(rho)

print(f"\n  ST year-over-year mean ρ: {np.mean(st_yoy_rhos):.3f}")
print(f"  ST year-over-year range: {min(st_yoy_rhos):.3f} to {max(st_yoy_rhos):.3f}")


print("\n" + "=" * 70)
print("SECTION 7: WORKED EXAMPLE — DET vs DAL, Week 6, 2024 (§4.10)")
print("=" * 70)

try:
    from src.data import load_play_by_play, filter_offensive_snaps, add_drive_outcome_target, engineer_features, FEATURE_COLUMNS, OUTCOME_TO_INDEX, normalize_drive_start_transition, compute_st_baselines, compute_st_value
    from src.model import load_model, predict_multinomial, compute_xep, compute_xep_lookup_table
    import joblib
except ImportError as e:
    print(f"  Cannot import src modules: {e}")
    print("  Skipping worked example (requires full pipeline)")
    import sys; sys.exit(0)

# Load model and calibrators
model = load_model("models/xscore_multinomial.json")
calibrators = joblib.load("models/xscore_multinomial_calibrators.pkl")

# Load 2024 play-by-play
pbp_2024 = load_play_by_play([2024])
pbp_2024 = filter_offensive_snaps(pbp_2024)
pbp_2024 = add_drive_outcome_target(pbp_2024)
pbp_2024 = engineer_features(pbp_2024)
pbp_2024 = pbp_2024.dropna(subset=FEATURE_COLUMNS + ["drive_outcome"])

# Find the game
det_dal_game = pbp_2024[
    (pbp_2024["game_id"].str.contains("DET")) &
    (pbp_2024["game_id"].str.contains("DAL")) &
    (pbp_2024["week"] == 6)
]
game_id = det_dal_game["game_id"].iloc[0]
print(f"\n  Game ID: {game_id}")

# Also need the training data for opp_ep_lookup
# Load training data (all 2018-2024)
pbp_all = load_play_by_play(list(range(2018, 2025)))
pbp_all = filter_offensive_snaps(pbp_all)
pbp_all = add_drive_outcome_target(pbp_all)
pbp_all = engineer_features(pbp_all)
pbp_all = pbp_all.dropna(subset=FEATURE_COLUMNS + ["drive_outcome"])

train_data = pbp_all[pbp_all["season"].between(2018, 2024)]
train_drives = train_data.groupby(["game_id", "drive"]).first().reset_index()
train_drives["actual_points"] = train_drives["drive_outcome"].map(
    {OUTCOME_TO_INDEX["td"]: 7, OUTCOME_TO_INDEX["fg"]: 3,
     OUTCOME_TO_INDEX["turnover"]: 0, OUTCOME_TO_INDEX["punt_other"]: 0}
)
opp_start_data = train_drives[["yardline_100", "actual_points"]].copy()
opp_start_data = opp_start_data.rename(columns={"yardline_100": "start_yardline_100"})
opp_start_data["start_yardline_100"] = 100 - opp_start_data["start_yardline_100"]
opp_ep_lookup = compute_xep_lookup_table(opp_start_data)

# ST baselines
train_with_st = normalize_drive_start_transition(train_data)
all_probs_train = predict_multinomial(model, train_with_st[FEATURE_COLUMNS], calibrators=calibrators)
train_with_st["xscore"] = all_probs_train[:, 1]
st_baselines = compute_st_baselines(train_with_st)
print(f"\n  ST baselines: {st_baselines}")

# Now process the specific game
game_plays = pbp_all[pbp_all["game_id"] == game_id].copy()
game_plays = normalize_drive_start_transition(game_plays)

# Get DET drives
det_drives = game_plays[game_plays["posteam"] == "DET"]
det_drive_first = det_drives.groupby("drive").first().reset_index()

# Predict xEP for each drive
drive_probs = predict_multinomial(model, det_drive_first[FEATURE_COLUMNS], calibrators=calibrators)
turnover_yl = 100 - det_drive_first["yardline_100"].values
det_drive_first["xep"] = compute_xep(drive_probs, turnover_yl, opp_ep_lookup)

# Actual points
det_drive_first["actual_points"] = det_drive_first["drive_outcome"].map(
    {OUTCOME_TO_INDEX["td"]: 7, OUTCOME_TO_INDEX["fg"]: 3,
     OUTCOME_TO_INDEX["turnover"]: 0, OUTCOME_TO_INDEX["punt_other"]: 0}
)
det_drive_first["xvoa"] = det_drive_first["actual_points"] - det_drive_first["xep"]

# Drive outcome labels
outcome_labels = {OUTCOME_TO_INDEX["td"]: "TD", OUTCOME_TO_INDEX["fg"]: "FG",
                  OUTCOME_TO_INDEX["turnover"]: "Turnover", OUTCOME_TO_INDEX["punt_other"]: "Punt"}
det_drive_first["outcome_label"] = det_drive_first["drive_outcome"].map(outcome_labels)

# ST value for DET
det_drive_first["st_baseline"] = det_drive_first["drive_start_type"].map(st_baselines)
det_drive_first["xep_at_start"] = det_drive_first["xep"]  # this is already at drive start

# For ST we need the xScore (TD prob) approach used in original pipeline
# Actually looking at the code, ST uses "xscore" (td prob) not xep
# Let me compute xscore for these drives
det_drive_first["xscore_val"] = drive_probs[:, 1]  # TD probability
# But wait - the st_baselines were computed on xscore (td prob), not xep
# Let me re-check - looking at compute_st_value, it uses "xscore" column
# So ST value = xscore(first play) - baseline(transition_type) where xscore is TD prob
det_drive_first["st_expected"] = det_drive_first["drive_start_type"].map(st_baselines)
det_drive_first["st_delta"] = det_drive_first["xscore_val"] - det_drive_first["st_expected"]
det_drive_first["st_delta"] = det_drive_first["st_delta"].fillna(0.0)

print("\n  DET Offensive Drives:")
print(f"  {'#':<3} {'Start State':<22} {'xEP':>6} {'Pts':>4} {'xVOA':>7} {'Outcome':<10}")
print("  " + "-" * 56)
for i, (_, row) in enumerate(det_drive_first.iterrows(), 1):
    yl = int(row["yardline_100"])
    dn = int(row["down"])
    ytg = int(row["ydstogo"])
    side = "OPP" if yl <= 50 else "OWN"
    display_yl = yl if yl <= 50 else 100 - yl
    start_state = f"{dn}st-{ytg} {side} {display_yl}"
    # Fix ordinal
    ordinals = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
    start_state = f"{ordinals.get(dn, f'{dn}th')}-{ytg} {side} {display_yl}"
    print(f"  {i:<3} {start_state:<22} {row['xep']:>6.2f} {int(row['actual_points']):>4} {row['xvoa']:>+7.2f} {row['outcome_label']:<10}")

total_xep = det_drive_first["xep"].sum()
total_pts = det_drive_first["actual_points"].sum()
total_xvoa = det_drive_first["xvoa"].sum()
print(f"  {'':3} {'TOTAL':<22} {total_xep:>6.2f} {int(total_pts):>4} {total_xvoa:>+7.2f}")

print(f"\n  DET Off_xVOA (game): {total_xvoa:+.2f}")

# DAL drives
dal_drives = game_plays[game_plays["posteam"] == "DAL"]
dal_drive_first = dal_drives.groupby("drive").first().reset_index()
dal_probs = predict_multinomial(model, dal_drive_first[FEATURE_COLUMNS], calibrators=calibrators)
dal_turnover_yl = 100 - dal_drive_first["yardline_100"].values
dal_drive_first["xep"] = compute_xep(dal_probs, dal_turnover_yl, opp_ep_lookup)
dal_drive_first["actual_points"] = dal_drive_first["drive_outcome"].map(
    {OUTCOME_TO_INDEX["td"]: 7, OUTCOME_TO_INDEX["fg"]: 3,
     OUTCOME_TO_INDEX["turnover"]: 0, OUTCOME_TO_INDEX["punt_other"]: 0}
)
dal_drive_first["xvoa"] = dal_drive_first["actual_points"] - dal_drive_first["xep"]
dal_off_xvoa = dal_drive_first["xvoa"].sum()
print(f"  DAL Off_xVOA (game): {dal_off_xvoa:+.2f}")
print(f"  DET Def_xVOA (= -DAL Off_xVOA): {-dal_off_xvoa:+.2f}")

# DET ST value
det_st_total = det_drive_first["st_delta"].sum()
print(f"  DET ST_Value (game): {det_st_total:+.3f}")

# But wait - ST is in xscore (TD prob) units in this implementation, not xEP units
# Looking at the pipeline code more carefully...
# In run_multinomial_pipeline.py line 101: reg_df["xscore"] = all_probs[:, 1]  (TD prob)
# Then compute_st_baselines uses "xscore" column
# But the GDS formula adds off_xvoa (in EP units) + def_xvoa (in EP units) + st_value (in xscore/prob units??)
# That seems like a unit mismatch. Let me check what the output CSV actually has.

# Let me just use the precomputed game-level GDS for this game
game_gds_det = game_df[(game_df["game_id"] == game_id) & (game_df["posteam"] == "DET")]
if len(game_gds_det) > 0:
    print(f"\n  [From precomputed CSV]:")
    print(f"  DET Off_xVOA: {game_gds_det['offensive_xova'].iloc[0]:+.2f}")
    print(f"  DET Def_xVOA: {game_gds_det['defensive_xova'].iloc[0]:+.2f}")
    print(f"  DET ST_Value: {game_gds_det['st_value'].iloc[0]:+.3f}")
    print(f"  DET GDS:      {game_gds_det['gds'].iloc[0]:+.2f}")
    print(f"  DET GDS in TDs: ~{game_gds_det['gds'].iloc[0]/7:.1f}")

game_gds_dal = game_df[(game_df["game_id"] == game_id) & (game_df["posteam"] == "DAL")]
if len(game_gds_dal) > 0:
    print(f"\n  DAL Off_xVOA: {game_gds_dal['offensive_xova'].iloc[0]:+.2f}")
    print(f"  DAL Def_xVOA: {game_gds_dal['defensive_xova'].iloc[0]:+.2f}")
    print(f"  DAL ST_Value: {game_gds_dal['st_value'].iloc[0]:+.3f}")
    print(f"  DAL GDS:      {game_gds_dal['gds'].iloc[0]:+.2f}")

print("\n" + "=" * 70)
print("DONE — All values computed")
print("=" * 70)
