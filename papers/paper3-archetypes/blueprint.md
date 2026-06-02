# Paper 3 — Outline

## Title

**"Offense Wins Championships: A Multi-Method Test of the Defensive Dominance Hypothesis in the Modern NFL"**

---

## Abstract (~250 words, draft)

The aphorism "defense wins championships" is perhaps the most widely cited empirical claim in professional American football, yet it has received remarkably little rigorous academic testing. We subject this claim to the first systematic, multi-method examination using a machine-learning-derived team quality decomposition. Using the xScore expected-points model and Game Deserved Score (GDS) framework (Author, 2026a; Author, 2026b), we decompose 224 team-seasons (2018-2024) into offensive and defensive expected value over average (xVOA) components and classify teams by archetype dominance. Three directional hypotheses operationalizing the defensive claim are tested via five statistical procedures: Spearman rank correlation, Pearson correlation with R² decomposition, binary logistic regression, OLS regression, and Cohen's d effect size estimation. All three hypotheses are rejected. Offense-dominant teams dramatically outperform defense-dominant teams in playoff advancement (Cohen's d = 0.672, 95% CI [0.574, 0.787]), offensive xVOA explains 3.3x more win-percentage variance than defensive xVOA (46.4% vs. 14.2%), and all seven Super Bowl champions in the study window were offense-dominant (offense share range: +0.581 to +0.923). No defense-dominant team won more than one playoff game across 94 playoff team-seasons. Robustness checks -- including opponent-strength controls, field-position mediation analysis, threshold sensitivity, and time-window variation -- confirm that the offensive advantage is structurally stable and not an artifact of schedule strength or measurement choices. We propose a "competent defense threshold" framework: offensive excellence drives championship contention while defensive adequacy serves as a necessary but insufficient secondary condition. These findings carry implications for salary-cap allocation strategy in constrained roster-construction environments.

---

## Section Structure with Word Budgets

### 1. Introduction (~1,500 words)

**Source:** Ch1 `sec-problem-statement` + `sec-research-gap` (extract RQ3 framing)

| Keep | Condense | Cut |
|------|----------|-----|
| Salary cap zero-sum framing (paras 1-2) | Rule-change history -> 1 paragraph | RQ1/RQ2 material (belongs to Papers 1 & 2) |
| "Bear Bryant" origin + cultural embedding | Existing metrics critique -> 2 sentences + cite Paper 1/2 | Thesis structure section |
| Stakes of misallocation argument | | Detailed EPA/DVOA/PFF critique (in Paper 1) |
| Research gap paragraph (final) | | |

**Guidance:**
- Open with salary-cap constraint + the phrase's cultural dominance (~400 words, draw from Ch1 paras 1-4, condense heavily)
- One paragraph on rule changes creating the modern offensive environment (~200 words, compress Ch1 para on 2004+ rule changes)
- Research gap: Robst et al. (2011) as only prior peer-reviewed study, its limitations in one paragraph (~300 words)
- "What's needed" -> cite Papers 1 & 2 as providing the measurement framework, state this paper applies it (~200 words)
- State RQ3 + three hypotheses clearly (~300 words -- can lift H1/H2/H3 verbatim from sec-hypotheses)

---

### 2. Related Work (~1,500 words)

**Source:** `sec-playoff-factors` (~2,900w) + `sec-resource-allocation` (~2,950w) + `sec-existing-work` (~3,000w)

| Keep | Condense | Cut |
|------|----------|-----|
| Robst et al. (2011) critique -- core paragraph | DWC literature overview -> 3 sentences | Baseball Pythagorean detail |
| Lock & Nettleton (2014) random forests | Salary cap as binding constraint -> 1 paragraph | Home-field advantage section (entire) |
| Massey & Thaler (2013) -- loser's curse core finding | Cross-sport lessons -> 2 sentences (Moneyball, 3-point revolution) | Era effects/rule changes (moved to Introduction) |
| Mulholland (2019) positional spending | Quantitative postseason determinants -> compressed | Borghesi (2008) -- internal pay equity |
| DVOA comparison (1 paragraph from sec-existing-work) | Lopez (2018) randomness -> 1 sentence | Basketball 3-point revolution detail |
| Novel contributions of this paper (from sec-existing-work, reframed) | Draft capital / loser's curse interaction -> 1 paragraph | Competitive strategy sustainability (too speculative) |
| | | GDS vs EPA detail (belongs to Paper 2) |
| | | Rathke (2017) xG parallel (in Paper 1) |

**Guidance:**
- **2.1 The "Defense Wins Championships" Debate** (~500 words): Robst et al. finding + limitations, note absence of modern follow-up, cultural persistence of the claim. Draw from sec-dwc-literature paras 1-2 + sec-comparison-robst.
- **2.2 Postseason Prediction and Randomness** (~300 words): Single-elimination variance (Lopez 2018), Lock & Nettleton's random forest finding that offensive variables dominate, need for multi-season aggregation. Compress sec-postseason-determinants.
- **2.3 Resource Allocation Under Salary Cap Constraints** (~500 words): Zero-sum framing, Mulholland (2019) positional spending, Massey & Thaler draft-capital interaction. Draw from sec-salary-cap + sec-positional-spending + sec-draft-capital, compress heavily.
- **2.4 Contribution of This Study** (~200 words): Four novel contributions (from sec-novel-contributions), reframed as paper contributions rather than thesis contributions. Cite Papers 1 & 2 for methodology.

---

### 3. Data and Methods (~2,200 words)

**Source:** `sec-archetype-analysis` (~3,100w), with 1-paragraph recaps of xScore and GDS citing Papers 1 & 2

| Keep (verbatim or near-verbatim) | Condense | Cut |
|------|----------|-----|
| H1, H2, H3 definitions (if not in Introduction) | xScore -> 1 paragraph recap + citation | Full xScore model description |
| offense_share formula (Eq. 1) | GDS framework -> 1 paragraph recap + citation | GDS derivation |
| IVs table (Tab 1) | Spearman/Pearson justification -> 1 sentence each | Detailed justification for Spearman over Pearson |
| DVs table (Tab 2) | OLS justification -> 2 sentences | Long paragraph on why OLS on ordinal is acceptable |
| Cohen's d formula + threshold definitions | Era comparison design -> 3 sentences (moved to results) | Full quartile analysis design rationale |
| Sample size + power considerations (compressed) | Quadrant analysis design -> 4 sentences | Detailed bye-week counting explanation |
| Logistic regression equation | | Holm-Bonferroni discussion (mention in results) |
| OLS regression equation | | |
| Opponent-strength control design | | |

**Guidance:**
- **3.1 Measurement Framework** (~300 words): One-paragraph xScore recap (multinomial drive-outcome model, 7 features, calibrated probabilities -> expected points). One-paragraph GDS recap (Off_xVOA, Def_xVOA, ST_Value decomposition). Cite Papers 1 & 2 for full details. Reader should understand "what it measures" without "how it works."
- **3.2 Sample and Variables** (~500 words): 224 team-seasons (32 x 7, 2018-2024). IVs table. DVs table. State offense_share formula explicitly: `offense_share = off_xvoa_per_game / (|off_xvoa_per_game| + |def_xvoa_per_game| + 0.01)`. Define defense-dominant threshold: offense share < -0.3 (with sensitivity range ±0.20 to ±0.40 tested in robustness). Keep epsilon regularization note as one sentence.
- **3.3 Hypotheses** (~200 words): If moved from Introduction, state H1/H2/H3 here. Otherwise cross-reference.
- **3.4 Statistical Procedures** (~800 words): Five tests, each in ~150 words. Keep equations for logistic + OLS. Cohen's d formula. Briefly note quartile analysis as non-parametric complement. Quadrant analysis design in 4 sentences.
- **3.5 Robustness Design** (~400 words): Opponent-strength control (leave-one-out), field-position mediation, threshold sensitivity, time-window variation, single-team exclusion. Each in 2-3 sentences stating what's tested and why.

---

### 4. Results (~3,500 words)

**Source:** `sec-archetype-findings` (~9,800w)

| Keep (verbatim or near-verbatim) | Condense | Cut |
|------|----------|-----|
| Quartile table (Tab: quartile-offense) | Descriptive quartile interpretation -> 1 tight paragraph | GDS-quartile table + full comparison (mention in 1 sentence) |
| SB participants table (Tab: sb-participants) | SB participant observations -> 2 paragraphs (from 4) | "Defense-Dominant Teams" subsection detail (fold into Cohen's d) |
| Statistical tests: all 5 core results | Era analysis -> 1 paragraph with table | KC 2023 "floor of offensive dominance" paragraph (move to Discussion) |
| Hypothesis summary table | Quadrant analysis -> 1 paragraph with table | PHI 2024 standalone paragraph (fold into SB table discussion) |
| Cohen's d result + CI | Robustness results -> 1 paragraph per check | "Negative defensive xVOA as structural pattern" (fold into SB discussion) |
| Spearman rho = 0.246 with CI | Sensitivity tables -> merge into 1 compact table | Full OLS opponent-control repetition (report once) |
| Pearson r = 0.681 / 0.376 with R² | Mediation -> compress to 4 sentences | Synthesis subsection (redundant -- hypothesis summary covers it) |
| Logistic regression table | | Second mention of OLS results |
| OLS results (beta_off = 0.099, beta_def = 0.065) | | Extended quartile interpretation paragraphs |
| Quadrant analysis table | | "Neither" quadrant extended discussion |
| Mediation result (9.2%) | | |
| Sensitivity: R² ratio across windows | | |
| Sensitivity: threshold variation table | | |

**Guidance:**
- **4.1 Descriptive Patterns** (~800 words): Quartile table + 1 paragraph interpreting the monotonic gradient. SB participants table + 2 paragraphs (all 7 offense-dominant; KC 2022 negative defense; range of shares). State headline: "no defense-dominant team won >1 playoff game."
- **4.2 Hypothesis Tests** (~1,200 words): Report all 5 tests sequentially.
  - Spearman: rho = 0.246, p = 0.0167, CI, interpretation (4 sentences)
  - Pearson: r = 0.681 / 0.376, R² ratio 3.3:1, interpretation (4 sentences)
  - Cohen's d: 0.672 [0.574, 0.787], means (0.60 vs 0.00), 0/29 defense-dominant won any PW (6 sentences)
  - Logistic: table, both significant, offense more precisely estimated (4 sentences)
  - OLS: beta coefficients, R² = 0.255, offense 50% larger (4 sentences)
  - Conclude with hypothesis summary table
- **4.3 Quadrant Analysis of Elite Teams** (~500 words): Table + key finding: Offense Only produced 4 SB winners (19.0%) vs Defense Only 2 (9.5%). Brief "Elite Both" note.
- **4.4 Robustness and Sensitivity** (~700 words):
  - Opponent-strength control: beta_3 = -0.001, p = 0.985 (3 sentences)
  - Field-position mediation: 9.2%, only ~1/10 of offensive advantage mediated (4 sentences)
  - Time-window sensitivity: R² ratio 2.5:1 to 4.0:1 across all windows (3 sentences + compact table)
  - Threshold sensitivity: 0 mean PW for defense-dominant at all thresholds +/-0.20 to +/-0.40 (3 sentences)
  - Single-team exclusion: KC 2023 removal doesn't change conclusions (2 sentences)
  - Era comparison: Spearman positive in both eras, Pearson offensive channel strengthens slightly (4 sentences)
- **4.5 Hypothesis Verdict** (~300 words): Summary table (keep verbatim from thesis). 3-sentence synthesis.

---

### 5. Discussion (~2,500 words)

**Source:** `sec-interpretation` (~4,100w) + `sec-implications` (~4,800w) + `sec-existing-work` positioning

| Keep | Condense | Cut |
|------|----------|-----|
| "What the Data Actually Show" core argument | Structural explanations -> 1 paragraph (from 3) | Full resource allocation section detail |
| Nuanced interpretation: offense gets you there, defense is the filter | Why the myth persists -> ~500 words (from ~1,100) | Draft strategy subsection (too prescriptive for journal) |
| Competent defense threshold concept | Strategic implications -> 1 paragraph of key insight | Coaching/organizational implications (too prescriptive) |
| KC 2023 reinterpretation | Comparison to Robst et al. -> 3 sentences | Implementation caveats section (fold into limitations) |
| 5/7 champions had negative def xVOA | | Moneyball/NBA parallels (in Related Work) |
| Structural explanations: variance asymmetry, rule environment, ceiling advantage | | Detailed cap-dollar recommendations |

**Guidance:**
- **5.1 Rejecting the Defensive Hypothesis** (~400 words): Summarize the verdict across all methods. Emphasize convergence. Draw from sec-verdict + sec-what-data-show (compress to strongest points only).
- **5.2 The Competent Defense Threshold** (~500 words): This is the paper's theoretical contribution beyond mere rejection. Offense as engine, defense as filter. KC 2023 as illustration (lowest offense share among winners, still offense-dominant). 5/7 champions with negative defensive xVOA. Draw from sec-nuanced-interpretation.
- **5.3 Structural Explanations** (~400 words): Why offense dominates -- three mechanisms in one paragraph each: (1) higher between-team offensive variance, (2) rule environment compresses defensive ceiling, (3) offense is proactive, defense reactive. Draw from sec-structural-explanations, compress each mechanism to 3-4 sentences.
- **5.4 Why the Myth Persists** (~300 words): Availability bias + denominator problem in 1 paragraph. Romantic appeal/confirmation bias in 1 paragraph. Keep Seattle 2013/Denver 2015 as single-sentence examples. Draw from sec-myth-persistence, cut extended discussion and final "refutation" paragraph.
- **5.5 Strategic Implications** (~400 words): One focused paragraph: salary cap as zero-sum -> 3.3:1 R² ratio implies asymmetric marginal returns -> offense-first allocation with competent-defense floor. Cite Mulholland (2019). State as conditional on associations being causal. Draw from sec-resource-allocation opening + sec-threshold-practice, heavily condensed. No detailed draft/coaching advice.
- **5.6 Limitations** (~450 words): (1) Seven-season window limits generalizability to this rule era (~80w), (2) no drive-level opponent adjustment in GDS, though OLS control was null (~80w), (3) correlation ≠ causation — observational data cannot prove investment causes wins (~100w), (4) logistic regression underpowered with 7 SB events / EPV = 3.5 (~80w), (5) era subsamples (3 seasons each) too small for formal inference on temporal trends (~60w), (6) QB quality as potential confound — elite QBs drive both offense share and playoff success (~50w). Draw from sec-limitations.tex + sec-sample-size power discussion.

---

### 6. Conclusion (~500 words)

**Source:** New writing, synthesizing

- Restate the central finding in one sentence
- Contribution to the literature: first multi-method test with calibrated ML decomposition
- Practical takeaway: offense-first with competent-defense floor
- Future work: longer time horizons, opponent adjustment, positional value decomposition, player-level xVOA (cite thesis future extensions)
- Closing line: "The evidence does not merely fail to support 'defense wins championships' -- it systematically contradicts it."

---

## Total Word Budget

| Section | Words |
|---------|-------|
| Abstract | 250 |
| 1. Introduction | 1,500 |
| 2. Related Work | 1,500 |
| 3. Data and Methods | 2,200 |
| 4. Results | 3,500 |
| 5. Discussion | 2,350 |
| 6. Conclusion | 500 |
| **Total** | **~11,800** |

Note: Trimmed §5.4 "Why the Myth Persists" by 200w and expanded §5.6 "Limitations" by 150w vs original. Net -50w from Discussion, keeping total under JQAS 12K ceiling.

Plus references, tables (5-6), and figures (2-3 from thesis figures). Well within JQAS 12,000-word upper bound.

---

## Key Condensing Decisions

1. **Era analysis** -> demoted from full subsection to 4 sentences in robustness (exploratory, underpowered, doesn't change conclusions)
2. **Quadrant analysis** -> kept but compressed (the Offense Only vs Defense Only comparison is a strong finding)
3. **Resource allocation lit** -> 1 paragraph of implications rather than full prescriptive section (JQAS is empirical, not advisory)
4. **"Why the Myth Persists"** -> kept at ~500 words (reviewers will ask "why does anyone believe this?" -- the availability bias argument answers it cleanly)
5. **Redundancy between descriptive + statistical results** -> merged into a single Results section where descriptive tables set up statistical tests rather than appearing as a separate "layer"
6. **GDS-quartile table** -> cut entirely (interesting but not central to the hypothesis; the offense-share quartile table is the one that matters)

---

## Tables to Include

1. Independent variables (from thesis Tab: ivs)
2. Dependent variables (from thesis Tab: dvs)
3. Offense-share quartile outcomes (from thesis Tab: quartile-offense)
4. Super Bowl participants with xVOA components (from thesis Tab: sb-participants)
5. Logistic regression output (from thesis Tab: statistical-tests)
6. Quadrant analysis of elite teams (from thesis Tab: quadrant-results)
7. Hypothesis testing summary (from thesis Tab: hypothesis-summary)

## Figures to Include

1. Offense share vs playoff wins scatter (thesis Fig: fig7_offense_share_pw)
2. Quartile bar chart (thesis Fig: fig4_quartile_bars)
3. Elite team quadrant diagram (thesis Fig: fig10_quadrant_diagram) -- optional, include if space permits

---

## Cross-Paper Citation Strategy

- Paper 1 (xScore): cite for model architecture, training procedure, calibration results, feature set
- Paper 2 (GDS): cite for decomposition methodology, validation against game outcomes, season-level stability
- Thesis (arXiv): cite for complete technical details, full robustness appendix, era analysis tables

This paper should be fully self-contained for a reader who accepts "xScore produces calibrated drive-outcome probabilities" and "GDS decomposes these into Off/Def/ST components" on trust + citation.
