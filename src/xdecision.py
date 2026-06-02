import numpy as np
import pandas as pd


def build_fg_pct_table(df: pd.DataFrame) -> pd.Series:
    """Build field goal success rate by yardline_100 from historical data.

    Filters to field goal plays, computes make rate by kick_distance, fills gaps
    using a ±3-yard smoothing window, then converts to a yardline_100-indexed
    Series (kick_distance = yardline_100 + 17).
    """
    fg_plays = df[df["play_type"] == "field_goal"].copy()
    fg_plays["made"] = (fg_plays["field_goal_result"] == "made").astype(int)
    pct_by_distance = fg_plays.groupby("kick_distance")["made"].mean()

    full_range = pd.Series(index=range(18, 70), dtype=float)
    for dist in full_range.index:
        if dist in pct_by_distance.index:
            full_range[dist] = pct_by_distance[dist]
        else:
            nearby = pct_by_distance[
                (pct_by_distance.index >= dist - 3) & (pct_by_distance.index <= dist + 3)
            ]
            full_range[dist] = nearby.mean() if len(nearby) > 0 else 0.0

    yardline_table = pd.Series(index=range(1, 100), dtype=float)
    for yl in yardline_table.index:
        kick_dist = yl + 17
        if kick_dist in full_range.index:
            yardline_table[yl] = full_range[kick_dist]
        else:
            yardline_table[yl] = 0.0
    return yardline_table


def build_ep_table(df: pd.DataFrame) -> pd.Series:
    """Build expected points by field position. EP = mean xScore at yardline * 7."""
    ep = df.groupby("yardline_100")["xscore"].mean() * 7
    return ep.sort_index()


def _conversion_probability(ydstogo: int) -> float:
    """Estimate 4th-down conversion probability based on distance."""
    if ydstogo <= 1:
        return 0.70
    elif ydstogo <= 3:
        return 0.55
    elif ydstogo <= 5:
        return 0.45
    elif ydstogo <= 10:
        return 0.35
    else:
        return 0.20


def _punt_net_yards(yardline_100: int) -> int:
    """Estimate average punt net yards from field position."""
    max_punt = min(40, yardline_100 - 10)
    return max(max_punt, 10)


def xdecision(
    yardline_100: int,
    ydstogo: int,
    ep_table: pd.Series,
    fg_pct: pd.Series,
) -> dict:
    """Compute expected points for Go, Kick, and Punt on 4th down."""
    p_convert = _conversion_probability(ydstogo)
    new_yardline = max(1, yardline_100 - ydstogo)
    opp_yardline_if_fail = 100 - yardline_100

    ep_if_convert = ep_table.get(new_yardline, 0)
    ep_opponent_if_fail = ep_table.get(opp_yardline_if_fail, 0) if opp_yardline_if_fail <= 99 else 0
    go_ep = p_convert * ep_if_convert - (1 - p_convert) * ep_opponent_if_fail

    fg_distance = yardline_100 + 17
    p_fg = fg_pct.get(yardline_100, 0) if fg_distance <= 63 else 0
    opp_yardline_if_miss = min(99, 100 - (yardline_100 + 7))
    ep_opponent_if_miss = ep_table.get(opp_yardline_if_miss, 0) if opp_yardline_if_miss <= 99 else 0
    kick_ep = p_fg * 3 - (1 - p_fg) * ep_opponent_if_miss

    punt_net = _punt_net_yards(yardline_100)
    opp_yardline_after_punt = min(99, 100 - (yardline_100 - punt_net))
    ep_opponent_after_punt = ep_table.get(opp_yardline_after_punt, 0) if opp_yardline_after_punt <= 99 else 0
    punt_ep = -ep_opponent_after_punt

    options = {"go": go_ep, "kick": kick_ep, "punt": punt_ep}
    recommendation = max(options, key=options.get)

    return {
        "go_ep": round(go_ep, 3),
        "kick_ep": round(kick_ep, 3),
        "punt_ep": round(punt_ep, 3),
        "recommendation": recommendation,
    }
