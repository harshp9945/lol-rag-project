# Feature Ablation Study

The ablation study measured where prediction power comes from by removing
feature families and re-training. The headline result: just three
features alone yield 0.90 AUC. This is a design finding rather than a
modelling result, showing that a tiny set of objective-control events
dominates match outcomes.

The full feature set contains engineered features including first
objectives, kill counts, advantage columns, interaction terms, and a gold
proxy.

TODO(from NB06): name the three features that alone reach 0.90 AUC and add
two or three ablation deltas.
