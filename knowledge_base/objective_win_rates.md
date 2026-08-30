# Objective Win Rates

Every major objective was tested for its effect on winning, using binomial
tests with 95% Wilson confidence intervals, chi-square tests with Cramer's
V for association strength, and a logistic regression (controlling for all
objectives at once) for odds ratios. All effects are statistically
significant (p < 0.001).

Win rate when a team secures each objective first:
- First Inhibitor: 91.1% (95% CI 90.8 to 91.4), large effect. The single
  strongest predictor of victory in the dataset.
- First Baron: 80.7% (95% CI 80.2 to 81.1), large effect.
- First Tower: 70.8% (95% CI 70.4 to 71.2), medium effect.
- First Rift Herald: 69.5% (95% CI 68.9 to 70.0), medium effect.
- First Dragon: 68.0% (95% CI 67.6 to 68.4), medium effect.
- First Blood: 59.1% (95% CI 58.7 to 59.5), small effect. Getting the first
  kill matters least of the objectives.

Association strength (chi-square, Cramer's V) ranks the same way: First
Inhibitor (V = 0.38) and First Baron (V = 0.29) dominate, while First Blood
(V = 0.01) is barely associated with the outcome despite being significant.

The logistic regression, which isolates each objective's independent
contribution, confirms the ranking by odds ratio: securing the first
inhibitor makes a team about 4.9 times more likely to win, first baron
about 1.55 times, first tower about 1.54 times, and first dragon about 1.44
times, all else equal. First blood adds the least (1.18 times).

The practical takeaway: structural and late-game objectives (inhibitor,
baron) decide games far more than early skirmish objectives (first blood).
