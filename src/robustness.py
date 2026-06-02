import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


def compute_opponent_adjustment(game_gds: pd.DataFrame) -> pd.DataFrame:
    """Compute leave-one-out opponent average GDS per team-season.

    Input must have columns: season, team, game_id, opponent, gds.
    For each team-season, computes the mean GDS of all opponents,
    where each opponent's GDS excludes the game played against the focal team.

    Returns DataFrame with columns: season, team, opp_avg_gds.
    """
    results = []

    for (season, team), team_games in game_gds.groupby(["season", "team"]):
        opp_gds_values = []
        for _, game in team_games.iterrows():
            opp = game["opponent"]
            game_id = game["game_id"]
            # Get opponent's GDS in all games EXCEPT the one against this team
            opp_other_games = game_gds[
                (game_gds["season"] == season)
                & (game_gds["team"] == opp)
                & (game_gds["game_id"] != game_id)
            ]
            if len(opp_other_games) > 0:
                opp_gds_values.append(opp_other_games["gds"].mean())

        opp_avg = np.mean(opp_gds_values) if opp_gds_values else 0.0
        results.append({"season": season, "team": team, "opp_avg_gds": opp_avg})

    return pd.DataFrame(results)


def _partial_correlation(x: pd.Series, y: pd.Series, control: pd.Series) -> float:
    """Compute partial correlation between x and y, controlling for control variable."""
    X_ctrl = sm.add_constant(control)
    resid_x = sm.OLS(x, X_ctrl).fit().resid
    resid_y = sm.OLS(y, X_ctrl).fit().resid
    r, _ = stats.pearsonr(resid_x, resid_y)
    return r


def compute_field_position_mediation(team_seasons: pd.DataFrame) -> dict:
    """Mediation analysis: Def_xVOA → avg_start_yardline → Off_xVOA → playoff_wins.

    Tests whether the offense-primacy finding survives after accounting for
    the defense→field position→offensive performance pathway.

    Input must have columns: off_xvoa_per_game, def_xvoa_per_game,
    avg_start_yardline, playoff_wins.

    Returns dict with correlation/partial-correlation statistics.
    """
    df = team_seasons.dropna(subset=[
        "off_xvoa_per_game", "def_xvoa_per_game", "avg_start_yardline", "playoff_wins"
    ])

    # Path a: Def_xVOA → avg_start_yardline
    r_def_fp, p_def_fp = stats.pearsonr(df["def_xvoa_per_game"], df["avg_start_yardline"])

    # Total effect: Off_xVOA → playoff_wins (zero-order)
    r_off_wins, p_off_wins = stats.pearsonr(df["off_xvoa_per_game"], df["playoff_wins"])

    # Partial correlation: Off_xVOA → playoff_wins, controlling for avg_start_yardline
    partial_r = _partial_correlation(
        df["off_xvoa_per_game"], df["playoff_wins"], df["avg_start_yardline"]
    )

    # Mediation percentage: how much of the zero-order r is explained by field position
    if abs(r_off_wins) > 1e-10:
        mediation_pct = max(0, (1 - abs(partial_r) / abs(r_off_wins)) * 100)
    else:
        mediation_pct = 0.0

    return {
        "def_to_field_pos_r": r_def_fp,
        "def_to_field_pos_p": p_def_fp,
        "off_xvoa_to_wins_r": r_off_wins,
        "off_xvoa_to_wins_p": p_off_wins,
        "off_xvoa_to_wins_partial_r": partial_r,
        "mediation_pct": mediation_pct,
    }


def run_controlled_regressions(df: pd.DataFrame) -> dict:
    """Run logistic and OLS regressions with and without opponent adjustment.

    Input must have: off_xvoa_per_game, def_xvoa_per_game, opp_avg_gds,
    made_playoffs, playoff_wins.

    Returns dict with model summaries for controlled vs uncontrolled.
    """
    results = {}

    X_base = df[["off_xvoa_per_game", "def_xvoa_per_game"]].copy()
    X_ctrl = df[["off_xvoa_per_game", "def_xvoa_per_game", "opp_avg_gds"]].copy()
    y_logit = df["made_playoffs"].astype(int)

    # Uncontrolled logistic
    X_unc = sm.add_constant(X_base)
    try:
        logit_unc = sm.Logit(y_logit, X_unc).fit(disp=0)
        results["logit_uncontrolled"] = {
            "off_coef": logit_unc.params.get("off_xvoa_per_game", 0),
            "off_pvalue": logit_unc.pvalues.get("off_xvoa_per_game", 1),
            "def_coef": logit_unc.params.get("def_xvoa_per_game", 0),
            "def_pvalue": logit_unc.pvalues.get("def_xvoa_per_game", 1),
            "pseudo_r2": logit_unc.prsquared,
        }
    except Exception:
        results["logit_uncontrolled"] = {"off_coef": 0, "off_pvalue": 1, "def_coef": 0, "def_pvalue": 1, "pseudo_r2": 0}

    # Controlled logistic
    X_c = sm.add_constant(X_ctrl)
    try:
        logit_c = sm.Logit(y_logit, X_c).fit(disp=0)
        results["logit_controlled"] = {
            "off_coef": logit_c.params.get("off_xvoa_per_game", 0),
            "off_pvalue": logit_c.pvalues.get("off_xvoa_per_game", 1),
            "def_coef": logit_c.params.get("def_xvoa_per_game", 0),
            "def_pvalue": logit_c.pvalues.get("def_xvoa_per_game", 1),
            "opp_coef": logit_c.params.get("opp_avg_gds", 0),
            "opp_pvalue": logit_c.pvalues.get("opp_avg_gds", 1),
            "pseudo_r2": logit_c.prsquared,
        }
    except Exception:
        results["logit_controlled"] = {"off_coef": 0, "off_pvalue": 1, "def_coef": 0, "def_pvalue": 1, "opp_coef": 0, "opp_pvalue": 1, "pseudo_r2": 0}

    # OLS: playoff_wins
    y_ols = df["playoff_wins"]

    X_unc_ols = sm.add_constant(X_base)
    ols_unc = sm.OLS(y_ols, X_unc_ols).fit()
    results["ols_uncontrolled"] = {
        "off_coef": ols_unc.params.get("off_xvoa_per_game", 0),
        "off_pvalue": ols_unc.pvalues.get("off_xvoa_per_game", 1),
        "def_coef": ols_unc.params.get("def_xvoa_per_game", 0),
        "def_pvalue": ols_unc.pvalues.get("def_xvoa_per_game", 1),
        "r_squared": ols_unc.rsquared,
    }

    X_c_ols = sm.add_constant(X_ctrl)
    ols_c = sm.OLS(y_ols, X_c_ols).fit()
    results["ols_controlled"] = {
        "off_coef": ols_c.params.get("off_xvoa_per_game", 0),
        "off_pvalue": ols_c.pvalues.get("off_xvoa_per_game", 1),
        "def_coef": ols_c.params.get("def_xvoa_per_game", 0),
        "def_pvalue": ols_c.pvalues.get("def_xvoa_per_game", 1),
        "opp_coef": ols_c.params.get("opp_avg_gds", 0),
        "opp_pvalue": ols_c.pvalues.get("opp_avg_gds", 1),
        "r_squared": ols_c.rsquared,
    }

    return results
