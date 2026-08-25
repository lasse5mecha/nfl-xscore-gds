# xScore & Game Deserved Score (GDS): NFL Team Quality Decomposition

A fully transparent, reproducible framework for measuring NFL team quality through drive-level expected-point modeling.

**xScore** is a calibrated multinomial XGBoost model predicting four-class drive outcomes (TD, FG, turnover, punt/other) from situational features. **GDS** decomposes team performance into three additive components---offensive xVOA, defensive xVOA, and special teams value---expressed on a common expected-points scale.

## Key Results

Current (2018--2025, 256 team-seasons, includes the 2025 season as an out-of-sample check on
the framework — see `papers/paper2-merged/`):

- Game-winner prediction accuracy: **85.7%** (vs. 57% home-team baseline)
- Season win-percentage variance explained: **74.3%** (r = 0.862, n = 256 team-seasons)
- Offense-to-defense explanatory ratio: **2.6:1** (never below 2.5:1 across five time-window checks)
- All 8 Super Bowl winners (2018--2025) are offense-dominant; no defense-dominant team-season has won more than one playoff game

## Papers

1. **xScore: A Calibrated Multinomial Model for Drive-Level Expected Points in the NFL**
   (`papers/paper1-xscore/`) — standalone, under review.
2. **Game Deserved Score: Decomposing NFL Team Quality and Testing the Defensive Dominance
   Hypothesis** (`papers/paper2-merged/`) — the active paper. Introduces GDS, validates it
   against game outcomes, then uses it to test the "defense wins championships" hypothesis.

`papers/paper2-gds/` and `papers/paper3-archetypes/` are earlier, separately-submitted drafts
of what is now the single `papers/paper2-merged/` manuscript; kept for historical reference
(see the `SUPERSEDED.md` note in each).

Extended version: Mecha, L. (2026). *xScore: A Machine Learning Framework for Evaluating NFL Team Performance and Playoff Success.* SSRN preprint. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6870618

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full multinomial pipeline (train + evaluate + compute GDS)
python run_multinomial_pipeline.py

# Extend the archetype/hypothesis-testing analysis to include a season
# (edit the season range in analyze_archetypes_v2.py, then:)
python analyze_archetypes_v2.py

# Regenerate every number reported in papers/paper2-merged/paper.tex
python papers/paper2-merged/compute_placeholders.py

# Run tests
pytest tests/
```

The pipeline downloads play-by-play data via `nfl_data_py`, trains the xScore model on 2018--2024 seasons, evaluates on the 2025 holdout, and computes GDS for all team-games. Outputs are saved to `output/`. Trained model weights and calibrators used to produce the numbers in the papers are included in `models/` so the archetype analysis and paper figures can be reproduced without retraining.

## Project Structure

```
src/
  data.py          Data loading, filtering, feature engineering, ST baselines
  model.py         Model training, calibration, xEP/xVOA/GDS computation
  robustness.py    Robustness checks and sensitivity analyses
  viz.py           Visualization (heatmaps, calibration plots, SHAP)
  xdecision.py     Decision-theoretic extensions (FG/punt expected value)

tests/             Unit and integration tests

models/            Trained xScore model (xscore_multinomial.json) and its
                    per-class isotonic calibrators
output/            Precomputed GDS/archetype datasets used by the papers

run_multinomial_pipeline.py    Main pipeline: train -> evaluate -> GDS
run_pipeline.py                Legacy binary (TD-only) pipeline
analyze_archetypes_v2.py       Archetype classification and hypothesis testing
generate_thesis_figures.py     Figure generation for papers
explore_archetypes.py          Exploratory archetype analysis
compute_bootstrap_cis.py       Bootstrap CIs for xScore model metrics
compute_gds_bootstrap_cis.py   Bootstrap CIs for GDS headline metrics

papers/paper2-merged/compute_placeholders.py   Regenerates every number
                                                reported in the active paper
```

## Data

All data are sourced from [nflfastR](https://github.com/nflverse/nflfastR) via the Python `nfl_data_py` wrapper. No proprietary data is required. The pipeline downloads data automatically on first run.

- **Training:** 241,195 plays from 2018--2024 (7 seasons)
- **Test:** 34,415 plays from 2025
- **Out-of-distribution:** 134,274 plays from 2014--2017

The 2025 test-set season is also used, out-of-sample relative to model training, as the
archetype-analysis extension in `papers/paper2-merged/` (256 team-seasons total).

## Model

The xScore model is a 4-class XGBoost classifier predicting drive outcomes from seven features:

| Feature | Description |
|---------|-------------|
| `down` | Current down (1--4) |
| `ydstogo` | Yards to first down/goal |
| `yardline_100` | Yards from opponent end zone |
| `score_diff` | Possession team score minus defense |
| `half_seconds_remaining` | Seconds left in current half |
| `goal_to_go` | Binary: yards to go equals yardline |
| `red_zone` | Binary: inside opponent 20 |

Team identity is deliberately excluded to produce a league-average baseline. Post-training isotonic calibration ensures predicted probabilities match observed frequencies (ECE = 0.010).

## GDS Framework

For each drive, expected points (xEP) are computed from the probability vector:

```
xEP = P(TD)*7 + P(FG)*3 + P(Punt)*0 - P(TO)*EP_opp(yardline)
```

Drive-level value over average: `xVOA = actual_points - xEP`

Game-level decomposition:
- **Off_xVOA:** Sum of xVOA across team's offensive drives
- **Def_xVOA:** Negative of opponent's Off_xVOA (mirror construction)
- **ST_Value:** Starting xEP deviation from transition-type baselines
- **GDS = Off_xVOA + Def_xVOA + ST_Value**

## License

MIT

## Citation

```bibtex
@misc{mecha2026xscore,
  author = {Mecha, Lasse},
  title = {{xScore}: A Machine Learning Framework for Evaluating {NFL} Team
           Performance and Playoff Success},
  year = {2026},
  howpublished = {SSRN preprint},
  doi = {10.2139/ssrn.6870618},
  url = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6870618}
}
```
