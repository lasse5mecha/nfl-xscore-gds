# Paper 2: Paragraph-Level Outline (v1)

Each entry: [¶ number] Key claim/content | ~word count | source

---

## 1. Introduction (1,200 words)

**¶1 — Point differential is noisy (250w)**
The most common summary of NFL game performance is point differential, but it is a poor measure of within-game quality. Scoring in football is discrete and lumpy — a single play can produce 7 points while an extended drive producing sustained field position may produce none. Approximately 25% of 2018–2024 regular-season games were won by the team that gained fewer yards and fewer first downs. Teams can lose the statistical battle while winning the scoreboard through turnover returns, short fields, or higher variance in timing. A team converting 3 of 8 FG opportunities is not straightforwardly inferior to one converting 2 of 2 TD opportunities, yet point differential treats them identically. This motivates a framework decomposing game-level performance into offensive, defensive, and special teams processes, evaluated in a common unit directly comparable across games.
*Source: sec-gds-framework.tex §1 (Motivation), moved to Introduction per blueprint.*

**¶2 — Existing metrics can't decompose reproducibly (300w)**
Three families of metrics attempt to measure team quality but none provides a transparent, reproducible three-component decomposition. EPA (nflfastR) offers a common currency for play comparison but aggregates noisily to the team level, lacks structured decomposition, and conflates game-script effects with quality. DVOA (Football Outsiders) provides three-component decomposition but is entirely proprietary — no external researcher has replicated its figures. PFF grades rely on subjective analyst judgment that is inherently non-reproducible. Win probability models attribute value to the team as a whole without distinguishing whether offense, defense, or special teams drove a favorable state. The absence of a transparent, verifiable team quality metric with explicit component decomposition constrains rigorous evaluation of questions about roster construction, strategic investment, and the relative importance of offensive versus defensive quality.
*Source: sec-nfl-metrics.tex, compressed. Ch1 lines 29–36.*

**¶3 — Contribution statement (350w)**
We introduce the Game Deserved Score (GDS), a fully transparent, reproducible framework decomposing NFL team quality into three additive components — offensive expected value over average (Off_xVOA), defensive expected value over average (Def_xVOA), and special teams value (ST_Value) — expressed on a common expected-points scale. GDS is built on drive-level expected-point predictions from a calibrated multinomial model (xScore; [cite Paper 1]) and evaluates each unit's contribution relative to league-average situational expectations. Across 1,848 regular-season games (2018–2024), the team with higher GDS wins 86.1% of contests (vs. 57% home-team baseline). At the season level, mean GDS/game explains 73.6% of win-percentage variance (r = 0.858, n = 224 team-seasons), with offense contributing 46.4% and defense 14.2% — a 3.3:1 ratio. A luck analysis demonstrates win-loss divergences of ±3–5 games from GDS-implied expectations, confirming that process-based measurement captures information beyond outcome-based records. The framework is the first open-source, three-component team quality metric suitable for cross-team comparison, strategic evaluation, and hypothesis testing in the modern NFL. No peer-reviewed work has previously provided a fully specified, publicly verifiable team quality decomposition at this level of analytical detail.
*Source: blueprint abstract + ch6 RQ2 answer.*

**¶4 — Roadmap (50w)**
The remainder of this paper describes the data and probability baseline, the GDS framework construction, four validation analyses, and implications for team quality measurement and strategic analysis.
*Source: Standard.*

---

## 2. Related Work (1,400 words)

**¶1 — EPA: different target, different purpose (250w)**
EPA assigns a point value to every game state and measures the change each play creates — answering "what is the expected next-score differential from this state?" It provides a common currency for comparing play types and is publicly available (nflfastR). EPA excels at its intended purpose: within-game play evaluation and marginal decision-making. However, GDS requires a different target entirely: calibrated drive-outcome probabilities estimating what *this drive* will produce for the possessing team, independent of future opponent scoring. The nflfastR EP model incorporates multi-possession scoring expectations, making it structurally unsuited as a denominator for single-drive quality measurement. Additionally, when EPA is aggregated to the team level, three systematic issues arise: play-level noise (single explosive plays dominate), within-drive attribution (cooperative sequences treated as independent), and game-script compression (leading teams' EPA suppressed by conservative play-calling). GDS uses xScore ([cite Paper 1]) as its baseline precisely because it provides calibrated drive-level outcome probabilities — a fundamentally different target from EPA. For the full distinction, see [cite Paper 1, §6.1].
*Source: sec-nfl-metrics.tex §1 + Paper 1 §6.1 "Different Targets" framing.*

**¶2 — DVOA: the reproducibility problem (350w)**
Football Outsiders' DVOA has been the most cited team quality metric for nearly two decades. Like GDS, it decomposes performance into offensive, defensive, and special teams components expressed as percentages relative to league average. DVOA adjusts for opponent quality (iterative recalibration) and game situation (down-weighting garbage time). Despite its sophistication, DVOA suffers from a fatal limitation for academic use: it is proprietary and non-reproducible. The precise opponent-adjustment algorithm, convergence properties, weighting function for garbage-time plays, normalization procedure, and handling of special situations have never been published. No external researcher has independently replicated DVOA's published figures. When DVOA identifies a team as the league's best defense, the claim rests entirely on authority rather than verifiable methodology. For questions involving hundreds of millions in cap spending — where the "defense wins championships" hypothesis directly informs decisions — relying on an unverifiable metric is scientifically unsatisfying. GDS addresses this gap: every component of the pipeline is fully specified and implemented in publicly available code.
*Source: sec-nfl-metrics.tex §2.*

**¶3 — PFF: subjectivity and non-reproducibility (200w)**
PFF employs trained analysts to grade every player on every play, producing player-level grades aggregated to teams. PFF captures process over outcome (correct reads graded independently of completion), which has genuine merit. However, the approach compounds the reproducibility problem: the grading is inherently non-reproducible across analysts, the weighting scheme converting play-level to season-level is proprietary, and there is no objective ground truth for validation. Two analysts watching the same play may assign different grades. For hypothesis testing requiring team quality decomposition on a common scale with quantifiable measurement properties, PFF is methodologically unsuitable.
*Source: sec-nfl-metrics.tex §3.*

**¶4 — Win probability models: can't decompose (200w)**
Win probability models estimate the likelihood of winning conditional on current game state, updating after every play. Lock & Nettleton (2014) demonstrated random forests outperform parametric logit for this task. WP models share xScore's commitment to situational conditioning but differ in target: WP predicts the binary game outcome while xScore predicts multinomial drive outcomes. The drive-level target enables the three-component decomposition that WP models cannot provide — a WP model attributes value to the team as a whole without distinguishing whether offense, defense, or special teams drove the favorable state. WP answers "how likely to win from here?" while GDS answers "how much better than average is this team's offense/defense performing?"
*Source: sec-nfl-metrics.tex §4.*

**¶5 — Gap statement and positioning (150w)**
No existing approach provides all four properties simultaneously: (1) calibrated probability baseline independent of team identity, (2) explicit three-component decomposition into offense/defense/special teams, (3) common expected-point scale enabling direct comparison across components, (4) fully open-source implementation enabling independent replication. EPA satisfies (4) but not (1–3). DVOA satisfies (2–3) but not (1) or (4). PFF satisfies none. GDS is designed to satisfy all four.
*Source: New synthesis from blueprint.*

**¶6 — Expected Possession Value analogy (100w)**
The central idea is analogous to Expected Possession Value decomposition in basketball analytics (Cervone et al., 2016): rather than asking "which team scored more?", the framework asks "which team generated and suppressed scoring-probability at a rate inconsistent with league expectations, and by how much?" The currency is expected points derived from a calibrated multinomial probability model applied at the drive level.
*Source: sec-gds-framework.tex lines 44–53.*

---

## 3. Data & Probability Baseline (600 words)

**¶1 — Data source and scope (150w)**
All data are sourced from nflfastR (Carl et al., 2021) via the Python nfl_data_py wrapper. The primary dataset spans 2018–2024 (7 seasons, 224 team-seasons, 241,195 filtered plays across 41,117 drives). Play filtering retains pass, rush, and sack plays; excludes kickoffs, punts, extra points, two-point conversions, kneels, spikes, and aborted snaps. The 2018 lower bound reflects the rule-change structural break in scoring distributions. Full data description and filtering criteria are reported in [cite Paper 1].
*Source: sec-data.tex, compressed per blueprint.*

**¶2 — Drive definition and target variable (100w)**
A drive begins when a team takes possession and ends at a terminal event: touchdown, field goal, turnover, punt, or end of half/game. The target is categorical: y ∈ {TD, FG, Turnover, Punt/Other}. Each play inherits its drive's terminal outcome. Defensive/special teams touchdowns are not credited to the possession team. The drive is the natural unit for team quality measurement because it encompasses the full cooperative sequence of plays within a single possession.
*Source: sec-data.tex §4.*

**¶3 — xScore recap and xEP formula (250w)**
The expected-points baseline is provided by xScore ([cite Paper 1]), a calibrated multinomial XGBoost model predicting P(TD|s), P(FG|s), P(TO|s), P(Punt|s) from seven situational features (down, ydstogo, yardline_100, score_diff, half_seconds_remaining, goal_to_go, red_zone). Team identity is deliberately excluded to produce a league-average baseline. The four-class probability vector converts to expected points via:

xEP(s) = 7·P(TD|s) + 3·P(FG|s) + 0·P(Punt|s) − EP_opp(yl)·P(TO|s)

where EP_opp(yl) is a field-position-dependent turnover penalty computed as the mean observed actual points across all training-window drives that began at the corresponding opponent yardline. The model achieves multiclass Brier 0.1562 on a held-out 2025 test set (BSS = 15.3% over naive baseline), with mean ECE of 0.010 confirming calibrated probabilities closely match observed frequencies. This calibration is essential: GDS treats xScore outputs as genuine expected values, so any systematic miscalibration would propagate into biased team quality estimates. Full model specification, calibration procedure, and evaluation are reported in [cite Paper 1].
*Source: Paper 1 results + sec-xscore-model.tex, condensed.*

**¶4 — Score differential conditioning (100w)**
A design feature critical for GDS validity: xScore conditions on score differential, so the baseline already accounts for game-script effects. A team protecting a 28-point lead receives lower xEP baselines (conservative play expected), meaning conservative drives that produce punts generate minimal negative xVOA. This conditioning addresses the game-script contamination that afflicts raw EPA aggregation and is a fundamental advantage of the GDS approach.
*Source: sec-nfl-metrics.tex last ¶ of §1.3.*

---

## 4. The GDS Framework (2,400 words)

**¶1 — Three-component decomposition overview (200w)**
GDS decomposes game-level deserved performance into three additive channels: Off_xVOA, Def_xVOA, and ST_Value. The decomposition is exhaustive: every drive routes through exactly one channel. Off_xVOA captures offensive execution above/below baseline. Def_xVOA captures defensive suppression of opponent offense. ST_Value captures field-position advantage conferred at drive start relative to transition-type baselines. All three are expressed in expected-point units. [GDS equation: GDS(A) = xVOA_Off(A) + xVOA_Def(A) + ST(A)]. A GDS of +7.0 means the team generated one full touchdown of net positive performance beyond league-average expectations in identical circumstances.
*Source: sec-gds-framework.tex §2.*

**¶2 — Offensive xVOA: drive-level formula + worked example (400w)**
Off_xVOA measures the net difference between points actually scored on a drive and the xEP at drive start. [pts(d) equation: 7 if TD, 3 if FG, 0 otherwise]. [xVOA drive equation: xVOA(d) = pts(d) − xEP(s_1(d))].

Full worked example: consider a drive beginning 1st-and-10 at the opponent's 30-yard line, with the model predicting xEP = 2.8 (reflecting ~30% TD probability, ~25% FG probability, and moderate turnover risk):
- Drive ends in TD: xVOA = 7 − 2.8 = +4.2 (substantial overperformance)
- Drive ends in FG: xVOA = 3 − 2.8 = +0.2 (modest overperformance — team exceeded expectation but only marginally)
- Drive ends in punt: xVOA = 0 − 2.8 = −2.8 (failure to convert a state with meaningful scoring expectation)
- Drive ends in turnover: xVOA = 0 − 2.8 = −2.8 (same penalty as punt for the offensive channel; field-position benefit to opponent enters through ST channel)

This graduated accounting — distinguishing 7-point, 3-point, and 0-point outcomes with the same situational denominator — is the primary advantage of the multinomial model over binary approaches. A binary model could only distinguish TD from not-TD, collapsing the FG/punt/turnover distinction entirely. Safeties classified as 0 points for possession team (opponent's 2-point gain captured through subsequent kickoff field position in ST channel). Safeties occur on <0.5% of drives.
*Source: sec-gds-framework.tex §3.1.*

**¶3 — Offensive xVOA: game-level aggregation (150w)**
Game-level Off_xVOA is the sum across all team drives in a game: [Off_xVOA(A) = Σ xVOA(d) for d in D(A)]. Interpretation: aggregate difference between actual points scored and expected points from league-average execution given the same sequence of drive-opening states. Positive = consistently converting above baseline; negative = failing to convert from high-xEP positions.
*Source: sec-gds-framework.tex §3.2.*

**¶4 — Defensive xVOA: mirror construction (250w)**
No separate defensive model is required. Def_xVOA(A) = −Off_xVOA(B). Logic: opponent's offensive xVOA measures points generated beyond situational expectation, which were generated against team A's defense. Negating converts this into a defensive penalty/credit. Two practical advantages: (1) guarantees zero-sum — Off(A) + Def(B) + Off(B) + Def(A) = 0 in every game; (2) avoids attribution ambiguity that arises if defensive xVOA were computed directly from defensive plays (sacks, TFLs, deflections) which overlap with offensive drive sequences and risk double-counting. Positive Def_xVOA = held opponent below expected output. Negative = allowed opponent to generate more above expectation than league-average defense would have.
*Source: sec-gds-framework.tex §4.*

**¶5 — Special Teams Value: transition types (300w)**
ST affects GDS through starting field position conferred at each drive. A kick returner advancing to the 35 instead of the 25 transfers 10 yards of field position → higher starting xEP. Drive transition types consolidated into 6 canonical categories for stable baseline estimation: KICKOFF, PUNT, MISSED_FG, INTERCEPTION, FUMBLE, DOWNS. [Table 1 — transition type taxonomy with raw subtypes]. Rare events (blocked kicks, muffed punts, onside recoveries) consolidated into parent categories matching typical starting field position. Consolidation rationale: rare types (blocked FGs occur <30 times per season league-wide) cannot support reliable mean xEP estimates; grouping with parent category ensures stable baselines. Each canonical category contains hundreds of drives in training window.
*Source: sec-gds-framework.tex §5.1.*

**¶6 — Special Teams Value: baseline and per-drive formula (300w)**
Expected starting xEP for each transition type estimated empirically: μ_τ = mean xEP at first play of all training-window drives of type τ. [Baseline equation]. Three design choices: (1) uses actual xEP at observed first play (not a synthetic neutral state), so baseline implicitly captures real game context in which each transition type tends to occur; (2) computed once on training window and held fixed for all downstream use (prevents leakage); (3) individual first-play xEP values reflect full situational state, so μ_τ integrates over the typical game states accompanying each type.

Per-drive ST value: ST(d) = xEP(s_1(d)) − μ_τ(d). [ST drive equation]. Positive = drive began from more favorable position than league average for that transition type. Negative = less favorable. A team with consistently good kick returns will systematically start KICKOFF drives above μ_KICKOFF, generating positive ST_Value.

Game-level: ST(A) = Σ ST(d) for all team drives. Rewards teams consistently securing superior starting positions across all drives regardless of mechanism (returns, coverage, opponent's failed fourth-down attempts).
*Source: sec-gds-framework.tex §5.2–5.4.*

**¶7 — Attribution rules: turnover handling (300w)**
A potential concern: could the same play contribute to multiple components? The GDS structure prevents double-counting. Turnover accounting (clearest illustration):
1. Offensive xVOA of turning team: drive ends at pts(d) = 0. xVOA = 0 − xEP(s_1) < 0. Full cost absorbed by turning team's offensive channel.
2. ST_Value of receiving team: subsequent drive begins with transition type INTERCEPTION/FUMBLE. Starting xEP compared to μ_INT or μ_FUM baseline. Field position credit goes to ST channel only.
3. Def_xVOA of receiving team: equals −Off_xVOA(turning team), which already includes the turnover. No additional defensive entry.
Result: clean partition. Cost borne by turning team's offense; positional benefit credited to receiving team's ST; defensive channel receives suppression credit without separate turnover entry. No play counted in more than one channel.
*Source: sec-gds-framework.tex §6.*

**¶8 — Formal properties (200w)**
Four properties supporting analytical use:
- Zero-sum: Off/Def contributions across both teams sum to zero in every game. ST_Value is NOT zero-sum (both teams can have positive ST).
- GDS(A) + GDS(B) = ST(A) + ST(B).
- Unified scale: all components in expected-point units. Off_xVOA of +1.0 = same advantage as ST_Value of +1.0.
- Additivity: game-level values sum to season totals. Decomposition preserved at every aggregation level.
- Decomposability: can isolate any single channel's contribution at per-game, per-week, per-season, or subset (playoff-only) levels.
*Source: sec-gds-framework.tex §7. Bullet list per blueprint.*

---

## 5. Validation (2,400 words)

**¶1 — Validation overview (100w)**
Four complementary analyses validate GDS as a meaningful measure of team-level competitive quality: game-level winner prediction, season-level correlation with win percentage, component structure of top/bottom teams, and luck analysis. Each addresses a distinct aspect of construct validity.
*Source: sec-gds-validation.tex intro.*

**¶2 — Game-winner prediction (400w)**
Across 1,848 regular-season games (2018–2024, excluding 7 ties), the team with higher GDS won 1,592 times → 86.1% accuracy. Baseline: home teams win ~57%. GDS improvement: 29.1 percentage points. The comparison is "noiseless" in the relevant sense: GDS is computed entirely from execution on that game day, without using final score or opponent identity as input.

The 13.9% divergence is a feature, not a bug: these are games where football's structural variance (turnover bounces, field goal efficiency, explosive-play variance) decoupled process from scoreboard. From GDS's perspective, these are games where the "wrong" team won probabilistically. Their existence confirms GDS is not re-encoding the final score but captures an orthogonal dimension of performance quality. This property — measuring who *deserved* to win rather than who *did* — is foundational.
*Source: sec-gds-validation.tex §1.*

**¶3 — Season-level correlation (500w)**
Unit of analysis: team-season (n = 224). Pearson r between mean GDS/game and win%: r = 0.858, p = 3.24 × 10⁻⁶⁶, R² = 0.736. GDS/game accounts for 73.6% of win-percentage variance. The remaining 26.4% is attributable to within-game turnover variance, opponent strength effects not absorbed into GDS, and genuine randomness in close-game resolutions.

[Figure 2 — GDS/game vs win% scatter].

Component decomposition [Table 2]:
- Off_xVOA/game: r = 0.681 [0.608, 0.743], R² = 46.4%
- Def_xVOA/game: r = 0.376 [0.260, 0.483], R² = 14.2%
- Combined GDS/game: r = 0.858 [0.824, 0.888], R² = 73.6%

Offense is the dominant predictor (46.4% individually). Defense adds distinct channel (14.2%). Combined R² (73.6%) exceeds sum of individual R² values (60.6%) by 13.0pp — synergistic structure: teams strong on both dimensions outperform additive expectation, justifying treatment of GDS as a unified index rather than a simple sum of independent components (cf. Robst et al., 2011 [cite robst2011defense]). The 3.3:1 offense-to-defense ratio in explained variance is the key structural finding.

[Figure 3 — Off/Def xVOA vs win% dual panel].
*Source: sec-gds-validation.tex §2.*

**¶4 — Component structure: top/bottom teams 2024 (500w)**
[Table 3 — top-5/bottom-5 by GDS/game 2024, with Off/Def xVOA and Win%].

Top 5: DET (+10.365), PHI (+8.384), BAL (+7.945), TB (+6.729), GB (+5.575).
Bottom 5: JAX (−5.835), NYG (−6.158), DAL (−6.259), CLE (−6.700), CAR (−8.211).

Four structural observations:
1. Offense dominates the top — all 5 derive majority from offense. DET: +12.980 off with −2.630 def. BAL: +11.035 off with −3.068 def. PHI is only top-5 team with positive def (+0.639).
2. Defense as complement, not engine — 4 of 5 top teams have negative Def_xVOA. Possible to be top-tier with below-average defense if offensive surplus is large enough.
3. Defense drives the floor — JAX, DAL, CAR have POSITIVE Off_xVOA but bottom-tier status from catastrophic defense. CAR: −10.852 Def_xVOA. Extreme defensive failure overrides positive offense.
4. Asymmetric variance — offensive range: −5.861 to +12.980 (spread 18.8). Defensive range: −10.852 to +0.639 (spread 11.5, skewed negative).
*Source: sec-gds-validation.tex §3.*

**¶5 — Luck analysis (400w)**
GDS-implied wins derived from cross-season regression. Gap between implied and actual = luck operationalization.

[Table 4 — Luckiest/unluckiest 2024]:
Luckiest: KC (15 actual vs ~10.2 implied, +4.8), HOU (10 vs ~6.7, +3.3), LA (10 vs ~6.9, +3.1).
Unluckiest: TEN (3 vs ~5.5, −2.5), TB (10 vs ~12.4, −2.4), NYJ (5 vs ~7.0, −2.0).

KC case most striking: moderate GDS/game (+2.92) yet 15 wins — largest positive divergence. Consistent with KC's documented fourth-quarter close-game conversion under Mahomes/Reid. TB: fourth-highest GDS but only 10 wins. These divergences reinforce the core motivation: W-L records are imperfect proxies for quality, and process-based measurement provides additional information.
*Source: sec-gds-validation.tex §4.*

**¶6 — Validation summary (100w)**
Four lines of evidence: 86.1% game-winner accuracy (vs 57% baseline), 73.6% season win% variance explained, structural diversity at top (multiple winning formulas), and ±3–5 game luck divergences. GDS is a validated measure of deserved competitive performance.
*Source: sec-gds-validation.tex §5.*

---

## 6. Discussion & Limitations (800 words)

**¶1 — What the 3.3:1 ratio means practically (200w)**
The offensive-to-defensive R² ratio of 3.3:1 has direct strategic implications: marginal investment in offensive quality yields markedly higher returns to winning than equivalent defensive investment. This does not mean defense is irrelevant — the component structure analysis shows extreme defensive failure (CAR 2024) can override positive offense. Rather, it means that once adequate defensive competence is secured, additional resources directed toward offense offer the higher expected payoff. The "defense wins championships" heuristic, while intuitively appealing, is not supported by the variance decomposition at the season level.
*Source: sec-implications.tex + blueprint.*

**¶2 — Comparison with existing metrics (200w)**
GDS achieves comparable discriminative validity to publicly reported DVOA correlations (DVOA reports r ≈ 0.85 with win%) while being fully reproducible. The advantage is not merely decomposition but the *calibrated* drive-level baseline — xScore's verified ECE of 0.010 ensures the denominators in the xVOA calculation match observed frequencies, a property no existing metric has demonstrated publicly. The 4-property advantage: (1) calibrated probability baseline independent of team identity, (2) three-component decomposition, (3) common expected-point scale, (4) open-source. Elo ratings capture quality but cannot decompose. PFF provides component insight but not on a common scale. GDS uniquely satisfies all four simultaneously. The trade-off: GDS does not adjust for opponent quality at the drive level (unlike DVOA), relying instead on large-sample averaging across 17 games to attenuate schedule effects.
*Source: blueprint + sec-existing-work.tex.*

**¶3 — Limitation: schedule strength (150w)**
GDS measures raw execution quality without drive-level opponent adjustment. A defense achieving high xVOA against below-average offenses is rated identically to one achieving the same xVOA against elite offenses. A leave-one-out opponent-strength control was included in OLS regression predicting playoff wins — coefficient was effectively zero (−0.001, p = 0.985), and Off/Def xVOA coefficients were unchanged. This null result provides empirical reassurance, though a principled iterative drive-level adjustment (analogous to Elo) would further strengthen internal validity.
*Source: sec-limitations.tex §3.*

**¶4 — Limitation: special teams construction (100w)**
ST_Value measures field-position deviation from transition-type baselines but does not independently model special teams scoring events. It also conflates any systematic bias in xScore baseline predictions with special teams value. A dedicated special teams probability model would provide a more principled decomposition but was beyond scope. The residual approach is acceptable given ST contributes a relatively small share of total game value.
*Source: sec-limitations.tex §6.*

**¶5 — Limitation: data scope (100w)**
The model conditions on publicly observable situational features only. Player tracking data, personnel groupings, and weather are absent. The 2018–2024 window coincides with offense-favoring rules. Periodic recalibration would be required if future rule changes rebalance the offensive-defensive equilibrium. The open-source design ensures longitudinal extension requires only addition of new season data.
*Source: sec-limitations.tex §4–5.*

---

## 7. Conclusion (400 words)

**¶1 — Summary of contribution (200w)**
We introduced GDS, the first fully transparent, reproducible team quality metric with explicit three-component decomposition for the NFL. Built on calibrated drive-level probability predictions from xScore [cite Paper 1], GDS expresses offensive, defensive, and special teams contributions on a common expected-point scale. The framework predicts game winners with 86.1% accuracy and explains 73.6% of season win-percentage variance (r = 0.858, n = 224 team-seasons). The component decomposition reveals offense as the dominant predictor (46.4% R²) with defense contributing a distinct but smaller channel (14.2%), yielding a 3.3:1 ratio in explanatory power.
*Source: ch6 RQ2 answer.*

**¶2 — Practical significance and future work (200w)**
GDS provides the analytical infrastructure for rigorous hypothesis testing about team quality, strategic investment, and the relative importance of different phases of play. In a companion paper [cite Paper 3], we apply the GDS component profiles to test the "defense wins championships" hypothesis through archetype classification and five-method statistical analysis. Extensions warranting investigation: (1) iterative drive-level opponent adjustment, (2) player-level attribution decomposing team xVOA to individuals, (3) salary-cap efficiency analysis matching positional spending to component contributions. The complete codebase, trained model, and data pipeline are publicly available for replication, critique, and extension. Full derivations, extended robustness analyses, and downstream applications are available in [cite arXiv thesis].
*Source: ch6 + blueprint.*

---

## Revised Word Budget

| Section | Target |
|---------|--------|
| Introduction | 1,200 |
| Related Work | 1,400 |
| Data & Baseline | 600 |
| GDS Framework | 2,400 |
| Validation | 2,400 |
| Discussion & Limitations | 800 |
| Conclusion | 400 |
| **Total** | **~9,200** |

---

## Figures & Tables

| # | Type | Content | Source file |
|---|------|---------|-------------|
| F1 | Figure | Pipeline flowchart | fig9_pipeline_flowchart.pdf |
| F2 | Figure | GDS/game vs win% scatter | fig6_gds_vs_winpct.pdf |
| F3 | Figure | Off/Def xVOA vs win% dual panel | fig3_xvoa_vs_winpct.pdf |
| T1 | Table | Drive transition type taxonomy (6 types) | — |
| T2 | Table | Component variance decomposition (r, CI, R²) | — |
| T3 | Table | Top-5/bottom-5 teams by GDS/game 2024 | — |
| T4 | Table | Luck analysis (luckiest/unluckiest 2024) | — |

---

## Key Numbers (must match)

- Games: 1,848 regular season (2018–2024, excluding 7 ties)
- Game-winner accuracy: 86.1%
- Home-team baseline: 57%
- Improvement: 29.1pp
- Pearson r (GDS/game vs win%): 0.858
- p-value: 3.24 × 10⁻⁶⁶
- R² combined: 73.6%
- Off_xVOA R²: 46.4%
- Def_xVOA R²: 14.2%
- Ratio: 3.3:1
- Synergistic surplus: 13.0pp (73.6% − 60.6%)
- Team-seasons: 224
- xScore Brier: 0.1562
- xScore BSS vs naive: 15.3%
- xScore ECE: 0.010
- KC luck divergence: +4.8 wins
- TB luck divergence: −2.4 wins

---

## Cross-Paper Citations

- Paper 1 (xScore): cite for model spec, calibration, SHAP, evaluation — referenced in §3 and §1
- Paper 3 (Archetypes): cite once in Conclusion — "In a companion paper, we apply..."
- Thesis (arXiv): cite once in Conclusion — "Full derivations available in..."
