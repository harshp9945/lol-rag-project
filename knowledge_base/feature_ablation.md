# Feature Ablation Study

The ablation study measured where prediction power comes from by testing
feature families in isolation. The headline result: just three features,
first inhibitor, first baron, and first tower, alone reach 0.9046 AUC.
Adding 27 more features and interaction terms lifts AUC by only about 0.09
(to 0.9975).

This is a data ceiling, not a model limitation. A tiny set of
objective-control events already determines almost the entire outcome, so
more features and more model complexity have little left to explain.

Feature-set comparison:
- Top 3 objectives (inhibitor + baron + tower): 0.905 AUC
- First objectives only (6 binary flags): 0.914 AUC
- Kill counts only: 0.949 AUC
- Advantage columns (per-objective Team 1 minus Team 2): 0.995 AUC
- Full 30-feature set: 0.9975 AUC

The practical read for a game team: matches are largely decided by who
controls structural objectives, which reinforces the objective-priority
findings from the strategy analysis.
