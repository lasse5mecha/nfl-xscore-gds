# Paper Alignment Document

## Status update (superseded trilogy structure)

Papers 2 (GDS) and 3 (Archetypes) were each rejected once (JSS: desk reject on fit; IJSF:
rejected because the `[Author, 2026]` self-citation to the companion paper had no matching
`references.bib` entry in the venue-anonymized submission copies, and because the SSRN working
paper's public abstract already reported paper 3's headline findings, making the "companion
paper" look like a repackaging of already-disseminated results rather than a distinct
contribution). Investigation traced both problems to the venue-specific anonymized copies
(`paper2-gds/jss/`, `paper3-archetypes/ijsf/`) — the master `references.bib` files always had a
correct, real SSRN entry for `mecha2026xscore`; the entry was dropped only when redacting the
in-text citation for blind review.

**Decision:** papers 2 and 3 are merged into a single manuscript, `papers/paper2-merged/`,
which cites the SSRN working paper openly (no redaction) and extends the sample by one full
season (2018--2025, 256 team-seasons) as a genuine out-of-sample check beyond what the working
paper reported. `papers/paper2-gds/` and `papers/paper3-archetypes/` are left in place as
source material and historical record — not deleted, not further modified for submission.

## Golden Rule
**DO NOT MODIFY any files in thesis/ or the root directory.** The arXiv thesis must remain intact. All paper work happens inside papers/.

---

## Structure

```
papers/
├── ALIGNMENT.md          ← This file (coordination)
├── paper1-xscore/         ← Standalone; under review at JQAS
│   ├── blueprint.md
│   └── paper1.tex
├── paper2-gds/             ← Superseded by paper2-merged/; kept as source material
├── paper3-archetypes/      ← Superseded by paper2-merged/; kept as source material
└── paper2-merged/         ← Active: GDS framework + defensive-dominance hypothesis test
    ├── blueprint.md      ← Section outline + word budgets + merge rationale
    ├── paper.tex         ← The actual paper
    ├── references.bib    ← Includes the real mecha2026xscore SSRN entry, cited openly
    ├── compute_placeholders.py  ← Regenerates every number in the paper from
    │                              output/archetype_v2_data.csv (256 rows) and
    │                              output/multinomial_game_gds.csv
    └── figures/
```

---

## Dependency Chain

```
Paper 1 (xScore, standalone) ──► Paper 2-merged (GDS + Archetypes)
     ▲                                    ▲
     └──── SSRN working paper cited openly by both ────┘
```

- Paper 1 is standalone, still under review at JQAS
- Paper 2-merged cites Paper 1 for the full model treatment and cites the SSRN working paper
  directly (real reference-list entry, third-person phrasing) rather than redacting it

---

## Shared Conventions

### Citation format
- SSRN working paper (Paper 2-merged, cited openly, no redaction): `\citep{mecha2026xscore}` —
  "Mecha (2026) develops..." (third person; never "as I show in my related work" — that
  phrasing is what breaks blind review, not the citation itself)
- Paper 1 from Paper 2-merged: `\citealt{mecha2026xscore-model}` — "The xScore model (Mecha,
  2026) provides..."
- The `mecha2026gds` and `mecha2026archetypes` placeholder keys (companion-paper stand-ins for
  the old paper2/paper3 split) are retired now that both are one manuscript; do not reintroduce
  them.

### Terminology (must be identical across all 3)
- Model name: xScore (italicized in running text)
- Framework name: GDS (Game Deserved Score)
- Component names: Off_xVOA, Def_xVOA, ST_Value
- Metric: xVOA (expected Value Over Average) — the final team-level metric
- Intermediate: xEP (expected points) — the per-drive probability-to-points conversion from xScore
- Distinction: xEP is the model output (per-drive expected points); xVOA is actual minus xEP (team quality signal)
- Unit: "expected-point units" (never "touchdown-probability units")
- Target: "drive outcomes" (4-class: TD, FG, turnover, punt/other)
- Offense share formula: off_xvoa_per_game / (|off_xvoa_per_game| + |def_xvoa_per_game| + 0.01)
- Defense-dominant threshold: offense share < -0.3 (default; sensitivity tested ±0.20 to ±0.40)

### Key numbers — xScore model (Paper 1; unchanged, training window still 2018–2024)
- Training plays: 241,195
- Test plays: 34,415
- OOD plays: 134,274
- Seasons: 2018–2024 (training), 2025 (test), 2014–2017 (OOD)
- Brier score: 0.1562 (test), 0.1529 (OOD)
- Naive baseline Brier: 0.1844 (test), BSS: 15.3%
- Logistic regression Brier: 0.1629 (test)
- Drive-clustered 95% CI: [0.1545, 0.1580]

### Key numbers — Paper 2-merged (GDS + archetype analysis, extended to 2018–2025)
Superseded 2018–2024/224-team-season numbers are struck through for reference; the pipeline
bug that produced them for Cohen's d (computing off/def-dominant means from `playoff_teams`
only instead of the full sample) was also fixed — see `analyze_archetypes_v2.py`.

- Team-seasons: 256 (2018–2025; ~~224 (2018–2024)~~)
- Playoff team-seasons: 108 (~~94~~)
- Super Bowl winners: 8, all offense-dominant (~~7~~)
- Game-winner accuracy: 85.7%, 2,119 games (~~86.1%, 1,848 games~~)
- GDS correlation with win%: r = 0.862, R² = 74.3% (~~r = 0.858, R² = 73.6%~~)
- Offense R²: 44.5%, Defense R²: 16.9%, Ratio: 2.6:1 (~~46.4% / 14.1% / 3.3:1~~) — ratio
  attenuates with the out-of-sample season added; never falls below 2.5:1 across five
  time-window checks (see paper §6.3 / robustness table)
- Spearman ρ: 0.208 (p = 0.0306), n = 108 playoff team-seasons (~~ρ = 0.246, p = 0.0167, n = 94~~)
- Cohen's d: 0.669 [0.580, 0.776], n_off = 166, n_def = 32 (~~0.672 [0.574, 0.787], n=143/29~~)
  — essentially unchanged despite the ratio attenuation; still zero playoff wins for any
  defense-dominant team-season
- Defense-dominant team-seasons: 32 (offense share < -0.3 threshold; ~~29~~)

### Spelling: American English
- rigor, judgment, modeled, center, defense, offense

---

## Cross-Cutting Decisions

*(The table below describes the original paper2/paper3 split and is kept for historical
reference — paper2-merged internalizes the "Paper 2" and "Paper 3" columns into one document,
so the repeats-vs-cites boundary between them no longer applies; only the Paper 1 boundary
still matters.)*

### What each paper repeats vs. cites

| Content | Paper 1 | Paper 2 | Paper 3 |
|---------|---------|---------|---------|
| nflfastR data source | Full (800w) | 1 paragraph + cite P1 | 1 sentence + cite P1 |
| xScore model | Full (2500w) | 1 paragraph recap | 1 sentence + cite P1 |
| xEP formula | Full derivation | Repeat equation only | Cite P2 |
| GDS construction | — | Full (1800w) | 1 paragraph recap |
| xVOA formula | — | Full derivation | Repeat equation only |
| Offense share | — | Define briefly | Full definition + thresholds |
| Calibration protocol | Full (500w) | 1 sentence | — |
| Isotonic regression | Full detail | — | — |
| SHAP analysis | Full (500w) | — | — |
| 5 statistical tests | — | — | Full (2000w) |
| Quadrant analysis | — | — | Full |
| Era analysis | — | — | Full (or appendix) |

### Data section per paper
- Paper 1: Full data description (800 words) — this is the reference
- Paper 2: "We use the dataset described in [cite P1]: 241,195 plays from 2018–2024..." (100 words)
- Paper 3: "Using the GDS framework [cite P2] applied to 224 team-seasons..." (80 words)

---

## Word Budget Summary

| Paper | Target | Status |
|-------|--------|--------|
| Paper 1 (xScore) | ~9,800 | Under review at JQAS |
| Paper 2-merged (GDS + Archetypes) | ~11,500–12,000 (raw `wc -w` currently ~8,700, comfortably under) | Draft complete, numbers verified against 256-team-season pipeline |

Word count for paper2-merged excludes an online supplement (luck analysis, point-differential
baseline, per-season/split-half reliability detail, full per-check robustness prose) that has
not been written as a standalone document yet — see `papers/paper2-merged/blueprint.md` for
what belongs there before submission.

---

## Submission Order
1. Paper 1 remains under review at JQAS (unaffected by this restructuring).
2. Paper 2-merged: target venue not yet finalized. JSS's own desk-reject suggested "a more
   technical performance analysis journal" — International Journal of Performance Analysis in
   Sport, Journal of Sports Economics, and JQAS itself (once paper 1's slot allows) are the
   live candidates. Re-submitting to IJSF is also worth considering: their rejection included
   detailed constructive feedback rather than a hard scope reject, and this draft directly
   addresses every specific critique they raised.

---

## Figures Allocation

| Figure (actual filename) | Paper 1 | Paper 2-merged |
|--------------------------|---------|----------------|
| fig1_calibration.pdf | ✓ | |
| fig2_field_heatmap.pdf | ✓ | |
| fig3_xvoa_vs_winpct.pdf | | ✓ |
| fig4_quartile_bars.pdf | | ✓ |
| fig5_shap_beeswarm.pdf | ✓ | |
| fig6_gds_vs_winpct.pdf | | ✓ |
| fig7_offense_share_pw.pdf | | ✓ |
| fig9_pipeline_flowchart.pdf | | ✓ |
| fig10_quadrant_diagram.pdf | | ✓ |

`fig8_era_comparison.pdf` is not used in paper2-merged — the era comparison is reported as a
compact table (robustness summary) rather than a figure, since the era-subsample evidence is
explicitly weaker in the 256-team-season sample than it was in the original 224-sample analysis
(see paper §7.3).
