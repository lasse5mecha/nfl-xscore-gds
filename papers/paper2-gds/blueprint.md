# Paper 2: Game Deserved Score — Decomposing NFL Team Quality into Offensive, Defensive, and Special Teams Value

**Target journal:** JQAS (~8,000–10,000 words)
**Status:** Outline
**Depends on:** Paper 1 (xScore) published separately

---

## Abstract (~200 words)

We introduce the Game Deserved Score (GDS), a fully transparent, reproducible framework that decomposes NFL team quality into three additive components—offensive, defensive, and special teams expected value over average (xVOA)—expressed on a common expected-points scale. GDS is built on drive-level expected-point predictions from a calibrated multinomial model (xScore; [cite Paper 1]) and evaluates each unit's contribution relative to league-average situational expectations. Across 1,848 regular-season games (2018–2024), the team with higher GDS wins 86.1% of contests. At the season level, mean GDS/game explains 73.6% of win-percentage variance (r = 0.858, n = 224 team-seasons), with offensive xVOA contributing 46.4% and defensive xVOA 14.2%—a 3.3:1 ratio in explanatory power. Component decomposition reveals that offensive xVOA exhibits higher between-team variance than defensive xVOA, providing greater discriminating power for team quality measurement. A luck analysis demonstrates win-loss divergences of ±3–5 games from GDS-implied expectations, confirming that process-based measurement captures information beyond outcome-based records. GDS provides the first open-source, three-component team quality metric suitable for cross-team comparison, strategic evaluation, and hypothesis testing in the modern NFL.

---

## Section Structure

| # | Section | Words | Source |
|---|---------|-------|--------|
| 1 | Introduction | 1,200 | ch1 (RQ2 framing) + new |
| 2 | Related Work | 1,400 | sec-nfl-metrics.tex (condensed) |
| 3 | Data & Probability Baseline | 600 | sec-data.tex + Paper 1 recap |
| 4 | The GDS Framework | 2,400 | sec-gds-framework.tex |
| 5 | Validation | 2,400 | sec-gds-validation.tex |
| 6 | Discussion & Limitations | 800 | sec-limitations.tex + ch6 |
| 7 | Conclusion | 400 | ch6 (RQ2 answer) |
| | **Total** | **~9,200** | |

Budget note: aim for 8,500 after editing passes. The 9,200 ceiling gives room to cut during polish.

---

## 1. Introduction (1,200 words)

**Purpose:** Motivate why point differential is noisy, why existing metrics can't decompose team quality reproducibly, and state the paper's contribution.

### What to keep/adapt from ch1-introduction.tex

- **Keep (condense to ~300 words):** The research-gap argument about EPA lacking decomposition, DVOA being proprietary, PFF being subjective (lines 29–36 of ch1). Compress the 4-property list into 2 sentences.
- **Keep (condense to ~200 words):** The RQ2 statement and what validation criteria GDS must satisfy (lines 53–58 of ch1).

### Write new (~700 words)

- A paper-specific opening about point-differential noise (draw from sec-gds-framework.tex §1 motivation, lines 4–30 — move it here as the paper hook). State the contribution clearly: "first fully transparent, reproducible 3-component team quality metric." End with a paragraph-level roadmap of the paper.

### What to cut from ch1

- All RQ1 and RQ3 material (belongs in Papers 1 and 3)
- The salary-cap framing, "defense wins championships" narrative, thesis structure section
- The historical discussion of Bear Bryant, media culture, etc.

---

## 2. Related Work (1,400 words)

**Purpose:** Position GDS against EPA, DVOA, PFF, and win-probability models. Establish the gap GDS fills.

### What to keep from sec-nfl-metrics.tex (~2,600 → 1,400 words)

- **Condense (~250 words):** EPA section — summarize the 3 limitations (noise, attribution, game-script) in 1 paragraph. Cite Paper 1 for full critique. Do NOT repeat the detailed 600-word argument (Paper 1 covers this fully).
- **Keep condensed (~350 words):** DVOA section (lines 33–46) — focus on the reproducibility problem. Cut the hedge about "the metric may well be excellent."
- **Keep condensed (~200 words):** PFF section (lines 48–57) — 1 paragraph on subjectivity + non-reproducibility.
- **Keep condensed (~250 words):** Win-probability section (lines 59–66) — 1 paragraph explaining why WP can't decompose.

### What to cut

- The detailed EPA formula explanation (JQAS readers know EPA)
- The Shapley value analogy (interesting but tangential)
- Extended PFF epistemological discussion

---

## 3. Data & Probability Baseline (600 words)

**Purpose:** Describe the dataset concisely and provide a 1-paragraph xScore recap so the paper is self-contained.

### What to keep from sec-data.tex (~2,450 → 400 words)

- **Keep condensed (~150 words):** Data source — nflfastR, 2018–2024, 224 team-seasons, ~241K plays. One sentence on why 2018 lower bound.
- **Keep condensed (~100 words):** Play filtering — just state what's included (pass, rush, sack) and excluded (special teams, kneels, spikes). Reference Paper 1 for full description.
- **Keep condensed (~100 words):** Target variable — the 4-class drive outcome. One sentence on why drive-level.
- **Cut entirely:** Feature engineering table, train/validation/test split details, rolling EPA shrinkage formula, full filtering table.

### xScore recap (write new, ~200 words)

Self-contained paragraph: "The expected-points baseline is provided by xScore ([cite Paper 1]), a calibrated multinomial XGBoost model that predicts P(TD), P(FG), P(Turnover), P(Punt/Other) from seven situational features. The four-class probability vector converts to expected points via xEP = 7·P(TD) + 3·P(FG). The model achieves [Brier score, calibration result]. Full specification and evaluation are reported in [Paper 1]."

State the xEP formula here (Eq. 1 of this paper) since GDS depends on it.

---

## 4. The GDS Framework (2,400 words)

**Purpose:** Core methodological contribution. Keep the mathematical rigor.

### What to keep from sec-gds-framework.tex (~2,800 → 2,400 words)

| Subsection | Action | Target words |
|------------|--------|--------------|
| §4.1 Motivation ("noisiness of point differential") | **Move to Introduction** — delete from here | 0 |
| §4.2 Three-Component Decomposition (lines 55–93) | Keep itemized definitions + GDS equation | ~400 |
| §4.3 Offensive Value Added (lines 110–199) | Keep drive-level xVOA formula, pts(d) definition, game-level aggregation, worked examples. Cut xG analogy and binary-vs-multinomial comparison. | ~600 |
| §4.4 Defensive Value Added (lines 201–240) | Keep mirror formula, zero-sum proof, "two practical advantages." Cut extended attribution-ambiguity prose. | ~350 |
| §4.5 Special Teams Value (lines 242–368) | Keep transition-type taxonomy table, baseline formula, per-drive ST formula, game-level aggregation. Cut extended "design choices" commentary. | ~550 |
| §4.6 Attribution / Turnover Handling (lines 375–417) | Keep the 3-step turnover accounting (key differentiator). Condense slightly. | ~300 |
| §4.7 Formal Properties (lines 419–477) | Keep zero-sum, unified scale, additivity, decomposability as compact bullet list. Cut proofs (state results only). | ~200 |

---

## 5. Validation (2,400 words)

**Purpose:** Four validation analyses. This is where the headline numbers live.

### What to keep from sec-gds-validation.tex (~4,900 → 2,400 words)

| Subsection | Action | Target words |
|------------|--------|--------------|
| §5.1 Game-Winner Prediction | Keep 86.1% figure, 57% baseline, 29.1pp improvement. Keep "13.9% divergence is a feature not a bug" paragraph. Cut "noiseless comparison" discussion. | ~500 |
| §5.2 Season-Level Correlation | Keep r=0.858, R²=0.736, p-value. Keep component decomposition table (Off 46.4%, Def 14.2%, combined 73.6%). Keep "synergistic structure" finding. Cut extended residual interpretation. | ~700 |
| §5.3 Component Structure Top/Bottom | Keep Table 2 (top-5/bottom-5) verbatim. Keep all four structural observations condensed to ~100 words each. | ~700 |
| §5.4 Luck Analysis | Keep Table 3 (KC, HOU, LA, TEN, TB, NYJ). Keep KC narrative (1 paragraph). Cut extended TEN/NYJ discussion. | ~400 |
| §5.5 Summary | One short paragraph synthesizing. | ~100 |

---

## 6. Discussion & Limitations (1,100 words)

### What to keep from sec-limitations.tex (GDS-specific only, ~500 words)

- **Keep condensed (~150 words):** Schedule strength — GDS doesn't adjust for opponent quality at drive level, but OLS control shows it doesn't matter.
- **Keep condensed (~150 words):** Special teams as residual — ST_Value conflates genuine ST impact with model error. Acknowledge, note it's small.
- **Keep condensed (~100 words):** Data limitations — no tracking data, no personnel groupings. One paragraph.

### What to cut (belongs in Paper 3)

- Correlation and causation
- Small championship sample
- Temporal confound

### Write new (~300 words)

- **Interpretation:** What does the 3.3:1 ratio mean practically? Offensive execution is the primary driver of winning, but defensive failure can override positive offense (Carolina example).
- **Comparison to existing metrics (~300 words):** Draw from sec-existing-work.tex — GDS achieves comparable discriminative validity to publicly reported DVOA correlations while being fully reproducible. Emphasize the 4 properties GDS satisfies that no single prior metric does (calibrated baseline, 3-component decomposition, common scale, open-source). Brief comparison to Elo and PFF on the same dimensions.
- **Future work teaser:** Opponent adjustment, player-level attribution, salary-cap efficiency.

---

## 7. Conclusion (400 words)

### What to keep from ch6-conclusion.tex

- **Keep condensed (~400 words):** The RQ2 answer (lines 18–24) — "First fully transparent, reproducible team quality metric with explicit three-component decomposition." Restate headline numbers. Note that GDS enables the archetype analysis reported in Paper 3. End with reproducibility commitment.

### What to cut

- RQ1 answer (Paper 1), RQ3 answer (Paper 3)
- Future research directions (covered in Discussion)
- Closing remarks about NFL decision-makers (Paper 3)

---

## Figures and Tables (6 total)

1. **Fig. 1:** Pipeline flowchart (fig9_pipeline_flowchart.pdf)
2. **Fig. 2:** GDS/game vs. win% scatter (fig6_gds_vs_winpct.pdf)
3. **Fig. 3:** Off/Def xVOA vs. win% dual panel (fig3_xvoa_vs_winpct.pdf)
4. **Table 1:** Drive transition taxonomy (compact version)
5. **Table 2:** Top-5/bottom-5 teams by GDS/game 2024
6. **Table 3:** Luck analysis (luckiest/unluckiest 2024)

Note: If hitting JQAS figure/table limit, fold the component-variance table into inline text.

---

## Cross-Paper References

| Cited work | What for |
|------------|----------|
| Paper 1 (xScore) | Model specification, calibration proof, feature engineering, SHAP analysis |
| Paper 3 (Archetypes) | Archetype classification that uses GDS components, hypothesis test results |
| Thesis (arXiv) | Full derivations, extended data description, robustness checks |

---

## Key Editorial Decisions

| Decision | Rationale |
|----------|-----------|
| Move "noisiness of point differential" from §4 to §1 | Better paper hook; avoids burying the lede in methodology |
| Cut xG/basketball analogies | JQAS readers know EPV; saves ~200 words |
| Keep all equations | Mathematical precision is the paper's differentiator vs. DVOA |
| Keep turnover attribution section | Unique contribution — shows no double-counting |
| Compress formal properties to bullet list | Saves ~200 words without losing substance |
| Keep luck analysis despite being tangential | Demonstrates GDS captures info beyond W-L; strong narrative examples |

---

## Source Files (thesis originals — DO NOT MODIFY)

- `thesis/chapters/ch3/sec-gds-framework.tex` (~2,800 words — CORE methodology)
- `thesis/chapters/ch4/sec-gds-validation.tex` (~4,900 words — CORE results)
- `thesis/chapters/ch3/sec-data.tex` (~2,450 words — shared, condense to ~400)
- `thesis/chapters/ch2/sec-nfl-metrics.tex` (~2,600 words — Related Work)
- `thesis/chapters/ch1-introduction.tex` (extract RQ2 framing ~800 words)
- `thesis/chapters/ch5/sec-limitations.tex` (GDS-specific limitations ~600 words)
- `thesis/chapters/ch6-conclusion.tex` (RQ2 answer ~400 words)

Total source: ~12,700 words → target ~8,000–9,000 words.
