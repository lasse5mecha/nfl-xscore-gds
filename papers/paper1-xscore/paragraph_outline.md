# Paper 1: Paragraph-Level Outline (v3 — post-second review)

Each entry: [¶ number] Key claim/content | ~word count | source

Changes from v2: Addressed 16 review points + 4 quick wins.

Critical fixes:
- Issue 1: Turnover penalty rewritten to match code (per-yardline lookup of observed points, not fixed constant)
- Issue 2: Naive baseline + logistic regression Brier added as comparators
- Issue 3: MACE removed (not computed in codebase); replaced with per-class ECE from calibration curves
- Issue 4: Paper 2 references reduced from 4 to 2; standalone utility strengthened with alternative applications

Structural fixes:
- Issue 5: EPA aggregation argument made precise (systematic bias, not noise); empirical comparison flagged as TBD
- Issue 6: Quantitative comparison acknowledged as non-apples-to-apples with explicit statement of why
- Issue 7: Phase 1 vs Phase 2 promoted to design-choice framing; overfitting explanation leads
- Issue 8: EPA critique compressed from 4 paragraphs (600w) to 2 paragraphs (300w); calibration literature expanded

Framing fixes:
- Issue 9: "Measurement instrument" framing replaces "application architecture"; no self-deprecation
- Issue 10: Salary cap opener replaced with proximate motivation
- Issue 11: Comparison table axes revised (merged open-source/reproducible, added trade-off axis)
- Issue 12: "Reproducibility" consolidated to single instance in Intro

Technical fixes:
- Issue 13: Train-on-all-plays vs apply-to-drive-start mismatch acknowledged
- Issue 14: Bootstrap CI noted as play-level; drive-clustered CI flagged as TBD
- Issue 15: Third OOD explanation (distribution shift) added
- Issue 16: Data section expanded to 1,000w with drive definition, garbage time, overtime, class imbalance

Quick wins applied:
- Drive count stated in Introduction
- Naive + logistic baseline Brier in Results
- "No peer-reviewed work..." sentence added to Related Work
- "Measurement instrument" replaces "application architecture" throughout

---

## 1. Introduction (1,100 words)

**¶1 — The measurement problem (200w)**
Evaluating whether a team's offense or defense drove its performance requires a calibrated probability baseline — a model estimating what a league-average team would produce from any given game state. This baseline must be independent of team identity so that deviations from expectation measure team quality rather than circular self-prediction. No such baseline currently exists in the public domain at the drive level. Existing approaches either operate at the play level (EPA), are proprietary (DVOA), or rely on subjective judgment (PFF). The absence of a transparent, calibrated drive-level model constrains downstream team quality measurement.
*Source: Ch1 paras 1-2, reframed. Issue 10 — no salary cap figure.*

**¶2 — EPA's limitations: systematic bias, not just noise (300w)**
EPA is the dominant public metric but has three systematic limitations when repurposed for team quality measurement: (1) play-level attribution within cooperative drive sequences — the value of any play depends on subsequent plays, creating non-independence that averaging cannot resolve, (2) game-script contamination — teams protecting leads adopt conservative play-calling that suppresses EPA regardless of team quality, distorting season-level aggregates, (3) outcome conflation — EPA produces a continuous value that conflates fundamentally different terminal events (touchdowns, field goals, turnovers, punts). Simply aggregating play-level EPA to the drive level reduces *random* variance but does not address these *systematic* biases: within-drive non-independence persists in the sum, game-script distortion affects every constituent play, and the continuous aggregation still conflates heterogeneous outcomes into a single number. What is needed is direct modeling of the drive outcome distribution — predicting what the drive will produce given its starting state, bypassing play-level attribution entirely. [TBD: If feasible, add 1-sentence empirical comparison of drive-aggregated EPA calibration vs xScore calibration on the same test set as evidence.]
*Source: Ch1 + Issue 5 — precise argument about systematic bias, not noise.*

**¶3 — Contribution statement (350w)**
We introduce xScore, a multinomial XGBoost model trained on 241,195 plays (~34,000 drives) from seven NFL seasons (2018–2024) that directly predicts four mutually exclusive drive outcomes (touchdown, field goal, turnover, punt/other) from seven situational features. The model converts these probabilities to expected points via a weighted formula with a field-position-dependent turnover penalty. Per-class isotonic calibration ensures predicted probabilities match observed frequencies. On a held-out 2025 test set, xScore achieves a multiclass Brier score of 0.156, representing a Brier skill score of 0.066 over a naive marginal-frequency forecaster (Brier 0.167). The model generalizes to pre-rule-change seasons (2014–2017) without degradation, learns a feature hierarchy consistent with domain knowledge (SHAP), and is released as a complete open-source codebase enabling independent replication and extension. Beyond serving as a probability baseline for team quality decomposition [cite Paper 2], xScore enables fourth-down decision analysis (comparing go/punt/kick expected values), drive-quality filtering for coaching evaluation, and in-game win-probability estimation grounded in calibrated outcome distributions. No peer-reviewed work has previously applied calibrated ensemble methods to direct multinomial drive-outcome prediction in the NFL.
*Source: Merged contribution + applications. Issue 4 — standalone utility strengthened. Issue 12 — single reproducibility mention.*

**¶4 — Roadmap (50w)**
The remainder of this paper describes the data and drive definition, model specification and calibration procedure, empirical results with baseline comparisons, and implications for sports analytics measurement.
*Source: Issue 9 — retained at 1 sentence.*

---

## 2. Related Work (1,400 words — reduced from 1,500)

**¶1 — xG in soccer: the methodological antecedent (250w)**
Expected Goals (xG) assigns each shot a goal probability conditional on situational features. The "actual minus expected" logic — separating what happened from what was expected — is the measurement architecture underlying our approach, transposed from shots in soccer to drives in American football. Key design principles adopted: exclude team identity from baseline (measure league-average expectation), operate at the natural unit of play (shots/drives rather than passes/plays), prioritize calibration over discrimination. Rathke (2017) established the framework with logistic regression; industry models (StatsBomb, Opta) moved to gradient boosted trees. Studies consistently find xG-based metrics outpredict raw goal counts for future performance, validating the "model the outcome distribution" approach.
*Source: sec-ml-sports §2, merged ¶1+¶2 from v2.*

**¶2 — EPA: the dominant NFL metric + its systematic limitations (300w)**
Expected Points Added assigns a point value to every game state and measures the change created by each play. It provides a common currency for comparing play types and is publicly available through nflfastR (Carl et al., 2021). However, three systematic limitations arise when EPA is repurposed for team quality measurement rather than play evaluation: (1) within-drive non-independence — plays within a drive form cooperative sequences where each play's value depends on subsequent trajectory, violating the independence assumption implicit in summation, (2) game-script contamination — teams with leads suppress EPA through conservative play-calling, creating systematic bias in season-level aggregation, (3) outcome conflation — the continuous value conflates fundamentally different terminal outcomes (a turnover and a punt produce similar EPA values despite different strategic implications).
*Source: sec-nfl-metrics §1. Issue 8 — compressed from 4 paragraphs to 2.*

**¶3 — Tree-based models in NFL analytics (200w)**
Lock & Nettleton (2014) demonstrated random forests could capture non-linear game-state relationships for NFL win probability, outperforming parametric alternatives. Yurko et al. (2019) applied hierarchical Bayesian models to EPA data for player-level evaluation. The nflfastR package democratized access to play-by-play data, enabling proliferation of public analytics — but no peer-reviewed work has applied ensemble methods to direct drive-outcome prediction, leaving a methodological gap between play-level EPA and team-level quality metrics.
*Source: sec-ml-sports §3, reframed as prior applications. Issue 6 restatement.*

**¶4 — Win probability models and DVOA/PFF (200w)**
WP models predict the binary game outcome from game state. While WP contributions can be attributed to possession side, the binary outcome structure prevents distinguishing offensive scoring mechanisms — it cannot separate how a team generates value (field goals vs touchdowns, turnover avoidance vs scoring efficiency) within any WP increment. DVOA provides three-component decomposition (offense, defense, special teams) but its methodology is proprietary and non-reproducible. PFF grades rely on subjective play-by-play judgment. Neither can be independently verified or extended.
*Source: sec-nfl-metrics §2-5, heavily compressed. Issue 8 — space recovered.*

**¶5 — Calibration in predictive sports models (300w)**
Niculescu-Mizil & Caruana (2005) demonstrated that boosted trees produce systematically miscalibrated outputs — overconfident near probability boundaries. For any metric treating predictions as genuine expectations (xG, xScore), miscalibrated probabilities propagate bias into every downstream calculation. Isotonic regression consistently outperforms Platt scaling for tree-based models due to non-parametric flexibility (Zadrozny & Elkan, 2002). The distinction between discrimination (ranking events by probability) and calibration (matching predicted to observed frequencies) is central to xScore's design: downstream team quality measurement requires calibration, not merely discrimination. A model that reliably ranks drives by TD probability but assigns 0.40 where the true rate is 0.25 would produce systematically biased team quality estimates when predictions serve as denominators.
*Source: sec-ml-sports §4. Issue 8 — expanded from 200w to 300w as calibration is methodological foundation.*

**¶6 — Gap statement (50w)**
No existing peer-reviewed approach provides: (1) drive-level multinomial outcome prediction, (2) per-class calibration verified against observed frequencies, and (3) open-source implementation enabling independent replication.
*Source: New synthesis. Quick win — explicit gap statement.*

---

## 3. Data (1,000 words — expanded from 800)

**¶1 — Data source (100w)**
All data are sourced from the nflfastR package (Carl et al., 2021), which provides play-by-play records for every NFL game since 1999. Data were accessed via the Python nfl_data_py wrapper. The package records ~45,000 plays per season with standardized variable names across years.
*Source: sec-data §1.*

**¶2 — Scope and sample (150w)**
Training: 2018–2024 (7 seasons, 241,195 filtered plays across ~34,000 drives). 2018 lower bound chosen because (1) 2018 rule changes materially altered scoring distributions (roughing-the-passer enforcement, lowered helmet rule), (2) the expanded 14-team playoff format is fully contained within this window, (3) data quality and play-type coding are substantially more reliable post-2018. Test: 2025 season (34,415 plays, ~4,900 drives, withheld from all modeling decisions). OOD: 2014–2017 (134,274 plays, ~19,000 drives, pre-rule-change stress test).
*Source: sec-data §2. Issue 16 — drive counts added.*

**¶3 — Drive definition (150w)**
A drive begins when a team takes possession and ends at a terminal event: score (touchdown, field goal), turnover (interception, fumble), punt, end of half, or end of game. Drives ending at the half or game end are classified by their terminal outcome (e.g., a drive ending in a field goal as time expires is coded as FG). Overtime drives follow the same classification. Drives of zero plays (e.g., muffed kickoff recovered for TD) are excluded. Each play inherits its drive's terminal outcome label: a 12-play touchdown drive assigns the TD label to all 12 constituent play observations.
*Source: sec-data, implicit in thesis. Issue 16 — now explicit.*

**¶4 — Play filtering (150w)**
Retained: pass plays, rush plays, sacks (classified as pass plays per standard practice). Excluded: kickoffs, punts (special teams plays), extra points, two-point conversions, QB kneels, spikes, aborted snaps. [Table 1 — filtering criteria with row counts]. The filtering preserves all plays representing offensive decision-making under standard football conditions while excluding ceremonial or special-teams actions.
*Source: sec-data §3. Keep table.*

**¶5 — Target variable and class imbalance (200w)**
The target is categorical at the drive level: y ∈ {TD, FG, Turnover, Punt/Other}. The play-level class distribution (TD 30.4%, FG 20.8%, Turnover 16.0%, Punt/Other 32.8%) overrepresents TDs relative to the drive-level rate (~20%) because touchdown-scoring drives contain more plays on average. This imbalance is handled by the log-loss objective (which penalizes confident wrong predictions regardless of class frequency) and stratified 5-fold cross-validation ensuring each fold maintains class proportions. No resampling or class weighting is applied. [Drive-outcome equation].
*Source: sec-data §4. Issue 16 — class imbalance strategy explicit.*

**¶6 — Features (100w)**
Seven features describe situational state: down, ydstogo, yardline_100, score_diff, half_seconds_remaining, goal_to_go, red_zone. [Table 2 — feature set with value ranges]. All features are publicly observable game-state variables; team identity is deliberately excluded to produce a league-average baseline.
*Source: sec-data §5. Keep table.*

**¶7 — Train/test split and garbage time (150w)**
Strict temporal split: 2018–2024 train, 2025 test. Random splitting rejected because play-level features (score_diff, half_seconds_remaining) within a season create temporal leakage between train/test. Hyperparameters selected via 5-fold temporally ordered CV within the training window using multiclass Brier score as criterion. Garbage time: no explicit filtering is applied. Score differential is included as a feature, so the model conditions on game state directly — drives in extreme score contexts produce appropriately low TD probabilities and high punt probabilities. This is preferable to arbitrary garbage-time thresholds that introduce researcher degrees of freedom.
*Source: sec-data §6. Issue 16 — garbage-time handling made explicit.*

---

## 4. Model (2,100 words — reduced from 2,200)

**¶1 — Design choice: Phase 1 vs Phase 2 (250w)**
A central design decision is whether the baseline should condition on team-quality features. A Phase 2 variant adds four team-quality inputs: rolling offensive EPA, rolling defensive EPA, within-game momentum, and home indicator. Phase 2 achieves marginally *worse* multiclass Brier (0.1589 vs 0.1562) on the held-out test set. The primary explanation is overfitting: rolling EPA estimates computed from ~50 prior games are noisy proxies of team quality, and fitting these noisy features degrades out-of-sample calibration. A secondary, complementary explanation: situational features implicitly encode team execution — strong teams systematically achieve more favorable down-and-distance states, so team quality is already partially captured by the distribution of game states a team reaches. The Phase 1 model (situational features only) is adopted as the primary specification. [Note: the 0.0027 Brier difference has not been tested for statistical significance via permutation test; we report it as directional evidence favoring the simpler model.]
*Source: sec-xscore-model §5. Issue 7 — promoted to lead, overfitting explanation first, significance caveat added.*

**¶2 — Model objective: probability estimator, not classifier (200w)**
xScore produces a four-class probability vector over mutually exclusive drive outcomes. [xScore equation]. The distinction from a classifier is substantive: probabilities serve as quantitative baselines for downstream team quality calculations, not merely rank-ordering devices. A classifier may achieve high accuracy with distorted probabilities; xScore requires accurate probabilities because downstream applications (team quality measurement, fourth-down analysis) divide actual outcomes by predicted expectations.
*Source: sec-xscore-model §1, first part.*

**¶3 — Multinomial vs binary justification (150w)**
A binary (TD vs no-TD) model conflates FGs, turnovers, and punts into a single failure category despite fundamentally different point values. A FG contributes 3 points; a turnover yields field position to the opponent. The multinomial model preserves this distributional information, enabling a principled expected-points calculation weighted by outcome-specific values.
*Source: sec-xscore-model §1, lines 24-33. Compressed slightly.*

**¶4 — The xEP formula + field-position-dependent turnover penalty (300w)**
Expected points: xEP(s) = 7·P(TD|s) + 3·P(FG|s) + 0·P(Punt|s) − EP_opp(yl_TO)·P(TO|s). [Equation]. The turnover penalty is field-position-dependent: for each starting yardline, we compute the mean *observed actual points scored* across all training-window drives that began at that position from the opponent's perspective (flipping the yardline: yl_opp = 100 − yl_current). This produces a lookup table mapping each yardline to the average points the opponent would score if given possession there. The approach avoids circularity by using observed outcomes (actual points: 7 for TD drives, 3 for FG drives, 0 otherwise) rather than model-predicted xEP — the lookup table is constructed entirely from empirical data, requiring no recursive call to the model itself. A turnover at the opponent's 20-yard line penalizes more heavily (~4.5 EP) than one at the opponent's 1-yard line (~0.5 EP), reflecting field-position reality. The unconditional mean across all turnover positions is approximately 2.0.
*Source: sec-xscore-model + model.py lines 281-313. Issue 1 — now matches code exactly.*

**¶5 — Calibration as primary objective (150w)**
For each class c, calibration requires P(outcome=c | predicted p) ≈ p for all p ∈ [0,1]. [Calibration condition]. This property is essential because downstream applications treat xScore outputs as genuine expected values. A model assigning 0.40 where the true rate is 0.25 would produce systematically biased quality estimates.
*Source: sec-xscore-model §1, lines 60-86.*

**¶6 — Architecture and hyperparameters (350w)**
XGBoost gradient boosted tree ensemble with multiclass log-loss (softmax) objective. Trees are constructed sequentially, each fitting the negative gradient of the loss. [Ensemble equation]. Two properties motivate tree-based methods: (1) the relationship between field position and TD probability is strongly non-linear (flat at midfield, inflecting sharply in the red zone) — trees partition this natively; (2) down×distance×field position interactions are captured at each node without analyst specification. Hyperparameters selected via grid search over temporally ordered 5-fold CV: n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8. [Table 3 — configuration].
*Source: sec-xscore-model §2+§3, compressed.*

**¶7 — Per-class isotonic calibration (200w)**
Despite probabilistic objectives, gradient boosted trees exhibit systematic over-confidence near probability boundaries (Niculescu-Mizil & Caruana, 2005). We apply per-class isotonic calibration: four independent non-parametric monotone functions map raw softmax scores to calibrated probabilities, then renormalize to unity. Isotonic regression is preferred over Platt scaling because it imposes no sigmoid functional form. The calibration uses out-of-fold predictions: each observation's calibration input was predicted by a model that never saw that observation during training, preventing optimistic bias.
*Source: sec-xscore-model §4.*

**¶8 — Evaluation metrics (250w)**
Three metrics assess complementary properties: (1) Multiclass Brier score — primary criterion, jointly rewards calibration and discrimination, strictly proper scoring rule. [Brier equation]. (2) Per-class AUC-ROC in one-vs-rest framework — discrimination quality, calibration-agnostic. (3) Calibration curves (reliability diagrams) — decile-binned comparison of predicted probability vs observed class frequency; perfect calibration lies on the 45° diagonal. Expected Calibration Error (ECE) quantifies the mean absolute deviation between predicted and observed frequencies across bins.
*Source: sec-xscore-model §6. Issue 3 — replaced MACE with ECE (which can be computed from existing calibration_curve output).*

**¶9 — Training vs application mismatch (150w)**
The model trains on all plays within drives (each inheriting the drive outcome label), giving it exposure to mid-drive states (e.g., 3rd-and-2 after two completions). However, for downstream xEP calculation, the model is applied only to the first play of each drive (the drive-start state). Calibration reported in Section 5 reflects all-play performance; calibration at drive-start states specifically may differ slightly because the training distribution includes mid-drive states that are systematically closer to terminal outcomes. We report overall calibration as the primary result and note this as a conservative assessment.
*Source: New writing. Issue 13 — acknowledged explicitly.*

---

## 5. Results (2,500 words — increased from 2,400)

**¶1 — Dataset summary (100w)**
Training: 241,195 plays across ~34,000 drives (2018–2024). Test: 34,415 plays across ~4,900 drives (2025). OOD: 134,274 plays across ~19,000 drives (2014–2017). All results reported on the 2025 test set unless stated otherwise.
*Source: sec-model-performance §1.*

**¶2 — Baseline comparisons and Phase comparison (300w)**
[Table 4 — model comparison: naive baseline, logistic regression, xScore Phase 1, xScore Phase 2, with Brier + CIs].
- Naive forecaster (always predicting marginal class frequencies [0.328, 0.304, 0.208, 0.160]): Brier = 0.167. This is the floor any useful model must beat.
- Multinomial logistic regression (same 7 features): Brier = [TBD — must compute].
- xScore Phase 1: Brier = 0.156 (bootstrap 95% CI [0.154, 0.158]).
- xScore Phase 2: Brier = 0.159 [0.156, 0.161].
Brier skill score (BSS) vs naive: (0.167 − 0.156) / 0.167 = 0.066 (6.6% improvement). Note: bootstrap CIs treat plays as independent; the effective sample size is smaller (~4,900 drives) due to outcome-label clustering within drives. Drive-clustered bootstrap [TBD — flag for implementation] would produce wider CIs.
*Source: New + sec-model-performance §2. Issues 2 + 14 — baseline added, clustering acknowledged.*

**¶3 — Per-class discrimination (200w)**
[Table 5 — per-class AUC-ROC with bootstrap CIs]. Punt/Other: 0.813 [0.806, 0.819], FG: 0.722 [0.713, 0.731], TD: 0.715 [0.707, 0.724], Turnover: 0.680 [0.669, 0.691]. Turnovers are the most stochastic outcome, consistent with their dependence on ball-carrier decisions and defensive opportunism that situational features cannot fully capture.
*Source: sec-model-performance §2, table.*

**¶4 — OOD generalization (300w)**
[Table 6 — OOD results]. OOD Brier: 0.150 [0.148, 0.152] — modestly better than the 2025 test set (0.156). Three non-exclusive explanations: (1) pre-2018 game states may be more prototypical (fewer unconventional play-calling patterns such as aggressive fourth-down attempts), (2) the 2025 test set may reflect continued offensive evolution creating distributional shift from the 2018–2024 training window, or (3) random variation between the two samples. We cannot distinguish these hypotheses from a single comparison. If explanation (2) is primary, it suggests periodic model retraining would be advisable as the game evolves — but the critical finding is that performance does not degrade catastrophically. The core functional relationships (field position, down) remain predictive across rule regimes.
*Source: sec-model-performance §3. Issue 15 — third explanation (distribution shift) explicitly added.*

**¶5 — Calibration results (250w)**
Post-isotonic calibration, reliability curves align closely with the 45° diagonal across all four classes. [Figure 1 — calibration curves, 4 panels]. Pre-calibration shows characteristic tree-ensemble over-confidence: raw softmax probabilities cluster toward extremes. Post-calibration corrects this. Expected Calibration Error (ECE) computed as the mean absolute deviation between predicted and observed bin frequencies: [TBD — compute from existing calibration curve output; estimated range 0.01–0.02 post-calibration vs 0.03–0.05 pre-calibration]. The calibration improvement directly contributes to the Brier score reduction.
*Source: sec-model-performance §4. Issue 3 — ECE replaces MACE, flagged as TBD computation.*

**¶6 — Intuition validation (350w)**
[Table 7 — intuition results with empirical base rates]. Four pre-specified scenarios compared against empirical base rates computed from the training data for matching game states:
(1) 1st-and-Goal from 1-yd: model 0.917, empirical rate 0.893 (n=412 drives). The model slightly overestimates, consistent with the goal-to-go indicator concentrating probability mass.
(2) 1st-and-Goal from 5-yd: model 0.799, empirical rate 0.734 (n=1,891). Above the empirical rate; the interaction between goal-to-go and short yardage pushes the estimate up.
(3) 1st-and-10 from own 30: model 0.256, empirical rate 0.213 (n=2,044). At the upper boundary; the neutral game-state conditioning of the validation scenario (0 score diff, full half remaining) makes it more favorable than the average drive from this position.
(4) 4th-and-22 at midfield: model 0.063, empirical rate 0.048 (n=89). Above empirical, though the small sample (89 drives) limits precision. Ordinal structure preserved: 0.917 > 0.799 > 0.256 > 0.063.
*Source: sec-model-performance §5.*

**¶7 — SHAP: field position dominates (150w)**
[Figure 3 — SHAP beeswarm]. yardline_100 has the largest mean absolute SHAP value. This is the primary structural check: a model failing to identify field position as dominant would have learned an incorrect probability surface regardless of aggregate metrics.
*Source: sec-model-performance §6, finding 1.*

**¶8 — SHAP: down is second, time/score are contextual (200w)**
Down ranks second. Time and score differential contribute at extremes only (two-minute drill, large differentials). Goal_to_go and red_zone show localized effects inside the opponent's 20. [Figure 2 — field heatmap]. The hierarchy — field position sets scale, down conditions opportunity, context modulates at margins — matches established domain knowledge.
*Source: sec-model-performance §6, findings 2-4.*

**¶9 — Summary (100w)**
Phase 1 achieves Brier 0.156 (BSS = 0.066 vs naive baseline), generalizes to pre-2018 eras, calibrates well across all classes, produces domain-consistent probability surfaces, and recovers the expected feature hierarchy. It is adopted as the primary model for downstream applications.
*Source: sec-model-performance §7.*

---

## 6. Discussion & Limitations (1,400 words)

**¶1 — The contribution: a measurement instrument for team quality (250w)**
While constituent methods (XGBoost, isotonic calibration, drive-level analysis) are individually well-understood, their integration into a drive-level measurement instrument solving the specific deficiencies of play-level EPA has not previously appeared in the peer-reviewed literature. The specific combination — multinomial four-class prediction at the drive level, per-class calibration to produce reliable expected-point baselines, team identity deliberately excluded — constitutes a purpose-built measurement instrument for team quality evaluation. This instrument solves the problem that play-level EPA aggregation cannot: it models what a drive *will produce* given its starting state, rather than summing what each play *did produce* after the fact.
*Source: Rewritten. Issue 9 — no apology, positive framing.*

**¶2 — Comparison table: xScore vs existing approaches (250w)**
[Table 8 — comparison on 5 axes].
Axes: (1) Publicly available / verifiable, (2) Calibrated (predicted ≈ observed frequencies), (3) Drive-level outcome model, (4) Distributional output (full probability vector, not point estimate), (5) Play-level granularity (available for individual plays).
- EPA (nflfastR): ✓ public, ✗ calibration not a design objective, ✗ play-level not drive, ✗ point estimate, ✓ play-level.
- DVOA (FO): ✗ proprietary, unknown calibration, ✗ not drive-level, ✗ index not distributional, ✗ proprietary.
- PFF: ✗ proprietary, ✗ subjective, ✗ not drive-level, ✗ ordinal scale, ✓ play-level.
- WP (nflfastR): ✓ public, partially calibrated, ✗ game-level binary, ✗ binary, ✓ play-level.
- xScore: ✓ public, ✓ calibrated, ✓ drive-level, ✓ 4-class distribution, ✗ drive-level only.
The trade-off: xScore sacrifices play-level granularity for drive-level calibration and distributional output. EPA retains play-level detail and broader community adoption.
*Source: Revised. Issue 11 — merged redundant axes, added honest trade-off (play-level granularity), replaced "decomposable" with "distributional output."*

**¶3 — Why not compare xScore and EPA on the same test set? (150w)**
A quantitative head-to-head comparison of xScore calibration vs drive-aggregated EPA calibration on identical held-out drives would be the most persuasive evidence for our claims. However, this comparison is not straightforward: EPA is defined per-play, so any drive-level aggregation requires researcher choices (sum? average? final-play value?) that introduce confounds. We note this as an important avenue for future work — a controlled study comparing xScore drive-outcome calibration against various EPA aggregation schemes on matched drives would strengthen the empirical case.
*Source: New writing. Issue 6 — honest acknowledgment.*

**¶4 — Temporal robustness (150w)**
The OOD result confirms that field position and down are structural features of American football rather than artifacts of a specific rule regime. If the superior OOD performance reflects distributional shift in 2025 (explanation 2 from Results), periodic retraining on updated data would maintain calibration — a practical consideration rather than a fundamental limitation.
*Source: Condensed.*

**¶5 — Downstream application (100w)**
In a companion paper [cite Paper 2], xScore predictions are aggregated into a Game Deserved Score framework for team quality decomposition. This represents one application; the calibrated probability output is equally suited to fourth-down decision modeling, drive-quality filtering, and in-game strategy optimization.
*Source: Issue 4 — reduced to one brief mention, emphasizes standalone utility.*

**¶6 — Limitation: data scope (100w)**
The model conditions only on publicly observable situational features. Player tracking data, personnel groupings, defensive coverage, and weather are absent. These omissions add noise but should not introduce systematic bias at the season level if effects are approximately symmetric across teams.
*Source: sec-limitations §4.*

**¶7 — Limitation: temporal scope (100w)**
The 2018–2024 training window coincides with an offense-favoring rule environment. The probability surface may be partially era-specific. Periodic recalibration on updated data would be required if future rule changes rebalance offensive-defensive equilibrium.
*Source: sec-limitations §5.*

**¶8 — Limitation: training distribution vs application (100w)**
The model trains on all plays within drives but is applied to drive-start states for xEP calculation. If calibration degrades at drive-start positions specifically (where the model has less information about eventual outcome than mid-drive), reported Brier scores may be optimistic for the xEP application. Future work should evaluate drive-start-only calibration explicitly.
*Source: New writing. Issue 13 — limitation version.*

**¶9 — Limitation: constant terms and approximations (100w)**
The field-position-dependent turnover penalty uses observed actual points rather than model-predicted values, avoiding circularity but introducing lookup-table approximation error for rare yardline positions with few observations. The 2025 test set bootstrap CI treats plays as independent despite outcome-label clustering within drives.
*Source: Issues 1 + 14 — limitations honestly stated.*

---

## 7. Conclusion (400 words)

**¶1 — Summary of contribution (150w)**
We introduced xScore, the first publicly available, calibrated, drive-level expected-points model for the NFL. The multinomial formulation predicts four drive outcomes and converts these to expected points via a field-position-dependent weighted formula. The model achieves Brier 0.156 (BSS = 0.066 vs marginal-frequency baseline), generalizes across eras (OOD Brier 0.150), and learns a domain-consistent feature hierarchy.
*Source: Synthesis.*

**¶2 — Practical significance (100w)**
By providing a transparent, calibrated probability baseline, xScore enables downstream team quality measurement, fourth-down analysis, and coaching evaluation independent of proprietary systems. The complete codebase, trained model, and data pipeline are publicly available for replication and extension.
*Source: Ch1 reproducibility argument. Issue 12 — final mention, earned.*

**¶3 — Future work (150w)**
Extensions: (1) drive-start-specific calibration evaluation, (2) context-dependent turnover penalty replacing the lookup table with a secondary model, (3) player-level attribution decomposing team-level value to individuals, (4) quantitative comparison with drive-aggregated EPA on matched test sets, (5) longer temporal windows as data accumulates. In a companion paper [cite Paper 2], we demonstrate xScore's application to team quality decomposition.
*Source: Ch6 future research, expanded with review-motivated items.*

---

## Revised Word Budget

| Section | v2 | v3 | Change |
|---------|----|----|--------|
| Introduction | 1,100 | 1,100 | — (tightened ¶2, expanded ¶3) |
| Related Work | 1,500 | 1,400 | -100 (EPA compressed, calibration expanded) |
| Data | 800 | 1,000 | +200 (drive def, garbage time, class imbalance) |
| Model | 2,200 | 2,100 | -100 (Phase 2 promoted, compressed elsewhere) |
| Results | 2,400 | 2,500 | +100 (baselines, OOD expansion) |
| Discussion | 1,400 | 1,400 | — (restructured, comparison table revised) |
| Conclusion | 400 | 400 | — |
| **Total** | **9,800** | **9,900** | **+100** |

Within JQAS ceiling. If tight, cut Intuition Validation to 250w (remove explanation text, keep table reference only).

---

## Tables & Figures (revised)

| # | Type | Content |
|---|------|---------|
| T1 | Table | Play filtering criteria |
| T2 | Table | Feature set (7 features + value ranges) |
| T3 | Table | Hyperparameter configuration |
| T4 | Table | **Model comparison: naive, logistic, xScore P1, xScore P2 (Brier + CIs)** |
| T5 | Table | Per-class AUC-ROC (+ CIs) |
| T6 | Table | OOD generalization (+ CIs) |
| T7 | Table | Intuition validation (with empirical base rates + n) |
| T8 | Table | **Comparison with existing approaches (5 axes, includes trade-off)** |
| F1 | Figure | Calibration curves (4 panels, pre+post) |
| F2 | Figure | TD probability heatmap by field position and down |
| F3 | Figure | SHAP beeswarm (TD class) |

---

## TBD Items (must resolve before writing)

1. **Logistic regression baseline Brier** — compute multinomial logistic regression on same features/test set
2. **ECE values** — compute from existing calibration_curve output (pre- and post-calibration)
3. **Drive-clustered bootstrap CI** — implement or explicitly state play-level CI as lower bound
4. **EPA vs xScore empirical comparison** — decide if feasible as 1-sentence result or deferred to future work
5. **Drive counts** — confirm exact numbers (~34,000 / ~4,900 / ~19,000) from pipeline output
