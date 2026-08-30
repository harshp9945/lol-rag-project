# Win Prediction Model Performance

The final win-prediction model achieves an AUC of 0.9977 under 5-fold
stratified cross-validation using HistGradientBoosting, compared against
Random Forest and GradientBoosting baselines.

The near-perfect AUC needs its most important caveat: the dataset
contains end-of-game objective totals, not mid-game state. The model is
therefore measuring that objectives CAUSE wins, not predicting outcomes
from incomplete information. This is a determinism finding about game
design, not a claim of real-time predictive power. A genuine real-time
predictor would use mid-game features such as gold difference at 15
minutes.

TODO(from NB06): add the AUC of the two baseline models and the top three
features by permutation importance.
