import numpy as np
import pandas as pd
from src.xdecision import build_ep_table, build_fg_pct_table, xdecision


def test_build_ep_table_covers_all_yardlines():
    df = pd.DataFrame({
        "yardline_100": list(range(1, 100)) * 100,
        "xscore": np.random.uniform(0, 1, 99 * 100),
    })
    ep_table = build_ep_table(df)
    assert len(ep_table) == 99
    assert ep_table.index.min() == 1
    assert ep_table.index.max() == 99
    assert all(ep_table >= 0)
    assert all(ep_table <= 7)


def test_build_ep_table_closer_yardlines_have_higher_ep():
    df = pd.DataFrame({
        "yardline_100": [5] * 100 + [80] * 100,
        "xscore": [0.7] * 100 + [0.05] * 100,
    })
    ep_table = build_ep_table(df)
    assert ep_table[5] > ep_table[80]


def test_xdecision_returns_all_three_options():
    ep_table = pd.Series(
        [6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5] + [1.0] * 89,
        index=range(1, 100),
    )
    fg_pct = pd.Series([0.95] * 30 + [0.7] * 20 + [0.4] * 49, index=range(1, 100))
    result = xdecision(
        yardline_100=5,
        ydstogo=3,
        ep_table=ep_table,
        fg_pct=fg_pct,
    )
    assert "go_ep" in result
    assert "kick_ep" in result
    assert "punt_ep" in result
    assert "recommendation" in result
    assert result["recommendation"] in ["go", "kick", "punt"]


def test_build_fg_pct_table_closer_is_higher():
    # kick_distance=20 → yardline_100=3, kick_distance=50 → yardline_100=33
    # 95% make rate at distance 20 vs 60% make rate at distance 50
    df = pd.DataFrame({
        "play_type": ["field_goal"] * 200,
        "kick_distance": [20] * 95 + [20] * 5 + [50] * 60 + [50] * 40,
        "field_goal_result": ["made"] * 95 + ["missed"] * 5 + ["made"] * 60 + ["missed"] * 40,
    })
    table = build_fg_pct_table(df)
    # table is indexed by yardline_100; yardline 3 = kick dist 20, yardline 33 = kick dist 50
    assert table[3] > table[33]
    assert 0 <= table[3] <= 1
    assert 0 <= table[33] <= 1


def test_xdecision_recommends_go_near_goal_line():
    ep_table = pd.Series(
        np.linspace(6.5, 0.3, 99),
        index=range(1, 100),
    )
    fg_pct = pd.Series([0.99] + [0.95] * 18 + [0.7] * 30 + [0.4] * 50, index=range(1, 100))
    result = xdecision(
        yardline_100=2,
        ydstogo=2,
        ep_table=ep_table,
        fg_pct=fg_pct,
    )
    assert result["recommendation"] == "go"
