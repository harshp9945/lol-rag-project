# Win-Rate Drivers (EXAMPLE — replace with real capstone output)

A feature ablation study measured each feature family's contribution to
win prediction by removing feature groups and re-training the model.

Objective control features (dragons, barons, towers) carried the largest
share of predictive signal, consistent with the game's macro-economy
design. Gold-difference features dominated importance rankings.

The final win-prediction model achieved high discrimination; the exact
AUC and validation protocol are documented in the capstone repository.
