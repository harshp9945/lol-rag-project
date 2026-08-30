# Win Prediction Model Performance

The final win-prediction model is a HistGradientBoosting classifier on 30
features. Under 5-fold stratified cross-validation it reaches an AUC of
0.9975 (standard deviation 0.0002), with a held-out test AUC of 0.9976 and
test accuracy of 97.4%. Precision and recall are both 0.97 for each team.

Model comparison across feature sets shows how little the fancy features
add once the core signal is present:
- First objectives only (6 binary flags): 0.914 AUC
- Kill counts only (5 features): 0.949 AUC
- Advantage columns only (Team 1 minus Team 2 per objective): 0.995 AUC
- All 15 baseline features: 0.997 AUC
- Fully expanded (30 features with interactions): 0.9975 AUC

The near-perfect AUC needs its most important caveat: the dataset contains
end-of-game totals, not mid-game state. The model is measuring that
objectives CAUSE wins, not predicting outcomes from incomplete
information. This is a determinism finding about game design, not a claim
of real-time predictive power. A genuine live predictor would need
mid-game features such as gold difference at 15 minutes.

By permutation importance, one feature dominates: tower kill advantage
(0.274), far ahead of inhibitor kills (0.015), objective dominance (0.013),
and first inhibitor (0.008). Grouped by type, advantage columns carry the
overwhelming majority of the predictive signal (0.279 total) versus kill
counts (0.027) and binary first-objective flags (0.015).
