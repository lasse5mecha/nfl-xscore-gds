# JQAS Paper Trilogy — Alignment Document

## Golden Rule
**DO NOT MODIFY any files in thesis/ or the root directory.** The arXiv thesis must remain intact. All paper work happens inside papers/.

---

## Structure

```
papers/
├── ALIGNMENT.md          ← This file (coordination)
├── paper1-xscore/
│   ├── blueprint.md      ← Section outline + word budgets
│   └── paper1.tex        ← The actual paper (to be written)
├── paper2-gds/
│   ├── blueprint.md      ← Section outline + word budgets
│   └── paper2.tex        ← The actual paper (to be written)
└── paper3-archetypes/
    ├── blueprint.md      ← Section outline + word budgets
    └── paper3.tex        ← The actual paper (to be written)
```

---

## Dependency Chain

```
Paper 1 (xScore) ──► Paper 2 (GDS) ──► Paper 3 (Archetypes)
     ▲                    ▲                    ▲
     └──── arXiv thesis cited by all three ────┘
```

- Paper 1 is standalone
- Paper 2 cites Paper 1 (1-paragraph xScore recap)
- Paper 3 cites Papers 1+2 (1-paragraph each recap)
- All three cite the arXiv preprint as "extended version"

---

## Shared Conventions

### Citation format
- arXiv thesis: `\cite{mecha2026xscore}` — "For full methodological details, see Mecha (2026)."
- Paper 1 from Paper 2: `\cite{mecha2026xscore-model}` — "The xScore model (Mecha, 2026a) provides..."
- Papers 1+2 from Paper 3: `\cite{mecha2026xscore-model,mecha2026gds}` — "Using the GDS framework (Mecha, 2026b), which builds on the xScore model (Mecha, 2026a)..."

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

### Key numbers (must match across all papers)
- Training plays: 241,195
- Test plays: 34,415
- OOD plays: 134,274
- Seasons: 2018–2024 (training), 2025 (test), 2014–2017 (OOD)
- Team-seasons: 224
- Playoff team-seasons: 94
- Super Bowl winners: 7 (all offense-dominant)
- Brier score: 0.1562 (test), 0.1529 (OOD)
- Naive baseline Brier: 0.1844 (test), BSS: 15.3%
- Logistic regression Brier: 0.1629 (test)
- Drive-clustered 95% CI: [0.1545, 0.1580]
- Game-winner accuracy: 86.1%
- GDS correlation with win%: r = 0.858, R² = 73.6%
- Offense R²: 46.4%, Defense R²: 14.2%, Ratio: 3.3:1
- Spearman ρ: 0.246 (p = 0.0167)
- Cohen's d: 0.672 [0.574, 0.787]
- Defense-dominant team-seasons: 29 (offense share < -0.3 threshold)

### Spelling: American English
- rigor, judgment, modeled, center, defense, offense

---

## Cross-Cutting Decisions

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
| Paper 1 (xScore) | ~9,800 | Blueprint complete |
| Paper 2 (GDS) | ~9,200 (aim 8,500 after polish) | Blueprint complete |
| Paper 3 (Archetypes) | ~11,200 (trimmed from 11,950) | Blueprint complete |
| **Total** | **~30,200** | |

Thesis total: ~42,000 words → Papers total: ~30,000 words (29% reduction)

JQAS ceiling: ~12,000 words body text. Paper 3 must stay under this after tables/figures are accounted for. Budget 500–800 words of cuts from "Why the Myth Persists" (§5.4) or move robustness details to online supplement.

---

## Submission Order
1. Upload arXiv thesis (once papers are outlined and any additions identified)
2. Submit Paper 1 to JQAS
3. Submit Paper 2 to JQAS
4. Submit Paper 3 to JQAS (headline paper, broadest audience)

All three to JQAS: the journal welcomes pure methods papers ("develop cutting edge methods") and the same-journal chain keeps cross-citations seamless and the reviewer pool consistent.

---

## Figures Allocation

| Figure (actual filename) | Paper 1 | Paper 2 | Paper 3 |
|--------------------------|---------|---------|---------|
| fig1_calibration.pdf | ✓ | | |
| fig2_field_heatmap.pdf | ✓ | | |
| fig3_xvoa_vs_winpct.pdf | | ✓ | |
| fig4_quartile_bars.pdf | | | ✓ |
| fig5_shap_beeswarm.pdf | ✓ | | |
| fig6_gds_vs_winpct.pdf | | ✓ | |
| fig7_offense_share_pw.pdf | | | ✓ |
| fig8_era_comparison.pdf | | | ✓ (or appendix) |
| fig9_pipeline_flowchart.pdf | | ✓ | |
| fig10_quadrant_diagram.pdf | | | ✓ (optional) |
