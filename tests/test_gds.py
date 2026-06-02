import pandas as pd
import numpy as np
from src.data import normalize_drive_start_transition


def test_normalize_maps_rare_types_to_parent():
    df = pd.DataFrame({
        "drive_start_transition": [
            "KICKOFF", "PUNT", "MUFFED_PUNT", "BLOCKED_PUNT",
            "INTERCEPTION", "FUMBLE", "DOWNS",
            "MISSED_FG", "BLOCKED_FG",
            "ONSIDE_KICK", "MUFFED_KICKOFF",
            "BLOCKED_PUNT,_DOWNS", "BLOCKED_FG,_DOWNS",
            "MUFFED_FG", "OWN_KICKOFF",
        ]
    })
    result = normalize_drive_start_transition(df)
    expected = ["KICKOFF", "PUNT", "PUNT", "PUNT",
                "INTERCEPTION", "FUMBLE", "DOWNS",
                "MISSED_FG", "MISSED_FG",
                "KICKOFF", "KICKOFF",
                "PUNT", "MISSED_FG",
                "MISSED_FG", "KICKOFF"]
    assert result["drive_start_type"].tolist() == expected


def test_normalize_preserves_nan():
    df = pd.DataFrame({
        "drive_start_transition": ["KICKOFF", None, "PUNT"]
    })
    result = normalize_drive_start_transition(df)
    assert result["drive_start_type"].iloc[0] == "KICKOFF"
    assert pd.isna(result["drive_start_type"].iloc[1])
    assert result["drive_start_type"].iloc[2] == "PUNT"


def test_normalize_keeps_original_column():
    df = pd.DataFrame({
        "drive_start_transition": ["MUFFED_PUNT", "KICKOFF"]
    })
    result = normalize_drive_start_transition(df)
    assert "drive_start_transition" in result.columns
    assert "drive_start_type" in result.columns


from src.data import compute_st_baselines


def test_compute_st_baselines_returns_dict_per_transition_type():
    df = pd.DataFrame({
        "drive_start_type": ["KICKOFF"] * 50 + ["PUNT"] * 50 + ["INTERCEPTION"] * 30,
        "xscore": [0.25] * 50 + [0.22] * 50 + [0.35] * 30,
        "drive": list(range(50)) + list(range(50)) + list(range(30)),
        "game_id": ["G1"] * 130,
        "posteam": ["KC"] * 130,
    })
    df["_is_first_play"] = True

    baselines = compute_st_baselines(df)

    assert "KICKOFF" in baselines
    assert "PUNT" in baselines
    assert "INTERCEPTION" in baselines
    assert abs(baselines["KICKOFF"] - 0.25) < 0.01
    assert abs(baselines["PUNT"] - 0.22) < 0.01
    assert abs(baselines["INTERCEPTION"] - 0.35) < 0.01


def test_compute_st_baselines_interception_higher_than_kickoff():
    """Interceptions start closer to endzone → higher avg xScore."""
    df = pd.DataFrame({
        "drive_start_type": ["KICKOFF"] * 100 + ["INTERCEPTION"] * 100,
        "xscore": [0.22] * 100 + [0.38] * 100,
        "drive": list(range(100)) + list(range(100)),
        "game_id": ["G1"] * 200,
        "posteam": ["KC"] * 200,
    })
    df["_is_first_play"] = True

    baselines = compute_st_baselines(df)
    assert baselines["INTERCEPTION"] > baselines["KICKOFF"]


from src.data import compute_st_value


def test_compute_st_value_positive_when_better_than_expected():
    """Drive starting closer to endzone than average → positive ST value."""
    df = pd.DataFrame({
        "game_id": ["G1", "G1", "G1", "G1", "G1", "G1"],
        "posteam": ["KC", "KC", "KC", "KC", "KC", "KC"],
        "drive": [1, 1, 1, 2, 2, 2],
        "drive_start_type": ["KICKOFF", "KICKOFF", "KICKOFF", "PUNT", "PUNT", "PUNT"],
        "xscore": [0.35, 0.40, 0.50, 0.45, 0.55, 0.60],
        "half_seconds_remaining": [1800, 1750, 1700, 1600, 1550, 1500],
    })
    baselines = {"KICKOFF": 0.25, "PUNT": 0.30}
    result = compute_st_value(df, baselines)

    kc = result[(result["game_id"] == "G1") & (result["posteam"] == "KC")]
    assert abs(kc["st_value"].iloc[0] - 0.25) < 0.01


def test_compute_st_value_negative_when_worse_than_expected():
    """Drive starting further from endzone than average → negative ST value."""
    df = pd.DataFrame({
        "game_id": ["G1", "G1"],
        "posteam": ["BUF", "BUF"],
        "drive": [1, 1],
        "drive_start_type": ["KICKOFF", "KICKOFF"],
        "xscore": [0.10, 0.15],
        "half_seconds_remaining": [1800, 1750],
    })
    baselines = {"KICKOFF": 0.25}
    result = compute_st_value(df, baselines)
    buf = result[(result["game_id"] == "G1") & (result["posteam"] == "BUF")]
    assert buf["st_value"].iloc[0] < 0


def test_compute_st_value_skips_unknown_transition():
    """Drives with unknown transition type get 0 ST value."""
    df = pd.DataFrame({
        "game_id": ["G1", "G1"],
        "posteam": ["KC", "KC"],
        "drive": [1, 2],
        "drive_start_type": ["KICKOFF", "UNKNOWN"],
        "xscore": [0.30, 0.50],
        "half_seconds_remaining": [1800, 1600],
    })
    baselines = {"KICKOFF": 0.25}
    result = compute_st_value(df, baselines)
    kc = result[(result["game_id"] == "G1") & (result["posteam"] == "KC")]
    assert abs(kc["st_value"].iloc[0] - 0.05) < 0.01


from src.model import compute_game_deserved_score, compute_xvoa


def test_gds_combines_three_components():
    """GDS = offensive xVOA - opponent offensive xVOA + ST value."""
    xvoa_df = pd.DataFrame({
        "game_id": ["G1", "G1"],
        "posteam": ["KC", "BUF"],
        "xVOA": [0.8, -0.3],
    })
    st_df = pd.DataFrame({
        "game_id": ["G1", "G1"],
        "posteam": ["KC", "BUF"],
        "st_value": [0.2, -0.1],
    })
    result = compute_game_deserved_score(xvoa_df, st_df)

    kc = result[result["posteam"] == "KC"].iloc[0]
    buf = result[result["posteam"] == "BUF"].iloc[0]

    # KC: off_xvoa=0.8, def_xvoa=-(-0.3)=+0.3, st=0.2 → GDS=1.3
    assert abs(kc["offensive_xvoa"] - 0.8) < 0.01
    assert abs(kc["defensive_xvoa"] - 0.3) < 0.01
    assert abs(kc["st_value"] - 0.2) < 0.01
    assert abs(kc["gds"] - 1.3) < 0.01

    # BUF: off_xvoa=-0.3, def_xvoa=-(0.8)=-0.8, st=-0.1 → GDS=-1.2
    assert abs(buf["offensive_xvoa"] - (-0.3)) < 0.01
    assert abs(buf["defensive_xvoa"] - (-0.8)) < 0.01
    assert abs(buf["st_value"] - (-0.1)) < 0.01
    assert abs(buf["gds"] - (-1.2)) < 0.01


def test_gds_higher_team_deserved_to_win():
    """Team with higher GDS deserved to win the game."""
    xvoa_df = pd.DataFrame({
        "game_id": ["G1", "G1"],
        "posteam": ["KC", "BUF"],
        "xVOA": [0.5, 0.2],
    })
    st_df = pd.DataFrame({
        "game_id": ["G1", "G1"],
        "posteam": ["KC", "BUF"],
        "st_value": [0.1, 0.05],
    })
    result = compute_game_deserved_score(xvoa_df, st_df)
    kc = result[result["posteam"] == "KC"].iloc[0]
    buf = result[result["posteam"] == "BUF"].iloc[0]
    assert kc["gds"] > buf["gds"]


def test_gds_works_with_multiple_games():
    xvoa_df = pd.DataFrame({
        "game_id": ["G1", "G1", "G2", "G2"],
        "posteam": ["KC", "BUF", "SF", "DAL"],
        "xVOA": [0.5, -0.2, 0.3, 0.1],
    })
    st_df = pd.DataFrame({
        "game_id": ["G1", "G1", "G2", "G2"],
        "posteam": ["KC", "BUF", "SF", "DAL"],
        "st_value": [0.1, -0.05, 0.0, 0.15],
    })
    result = compute_game_deserved_score(xvoa_df, st_df)
    assert len(result) == 4
    assert set(result.columns) == {"game_id", "posteam", "offensive_xvoa", "defensive_xvoa", "st_value", "gds"}


def test_full_gds_pipeline_integration():
    """End-to-end: plays → xVOA → ST value → GDS for a two-team game."""
    plays = pd.DataFrame({
        "game_id": ["G1"] * 12,
        "posteam": ["KC", "KC", "KC", "BUF", "BUF", "BUF",
                    "KC", "KC", "KC", "BUF", "BUF", "BUF"],
        "drive": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        "xscore": [0.20, 0.35, 0.55,
                   0.30, 0.25, 0.20,
                   0.40, 0.60, 0.75,
                   0.35, 0.30, 0.25],
        "drive_td": [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        "half_seconds_remaining": [1800, 1780, 1760, 1700, 1680, 1660,
                                   1500, 1480, 1460, 1400, 1380, 1360],
        "drive_start_type": ["KICKOFF", "KICKOFF", "KICKOFF",
                             "KICKOFF", "KICKOFF", "KICKOFF",
                             "PUNT", "PUNT", "PUNT",
                             "PUNT", "PUNT", "PUNT"],
    })

    # Compute xVOA
    xvoa_df = compute_xvoa(plays[["game_id", "posteam", "drive", "xscore", "drive_td", "half_seconds_remaining"]])
    assert len(xvoa_df) == 2

    # Compute ST value
    baselines = {"KICKOFF": 0.25, "PUNT": 0.30}
    st_df = compute_st_value(plays, baselines)
    assert len(st_df) == 2

    # Compute GDS
    gds_df = compute_game_deserved_score(xvoa_df, st_df)
    assert len(gds_df) == 2

    kc = gds_df[gds_df["posteam"] == "KC"].iloc[0]
    buf = gds_df[gds_df["posteam"] == "BUF"].iloc[0]

    # KC had two TD drives, BUF had two failed drives → KC should have higher GDS
    assert kc["gds"] > buf["gds"]
    # KC offensive xVOA should be positive (scoring drives)
    assert kc["offensive_xvoa"] > 0
    # BUF offensive xVOA should be negative (declining drives, no TDs)
    assert buf["offensive_xvoa"] < 0
