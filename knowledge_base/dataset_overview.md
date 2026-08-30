# Dataset Overview

The analysis covers 51,490 ranked League of Legends matches from Season 9
(June to September 2017), sourced from a public Kaggle ranked-matches
dataset.

Season 9 was chosen deliberately: it is a complete competitive season
with no mid-season patch disruptions, which makes it suitable for
statistical analysis without confounding meta shifts. Champion-specific
findings reflect that meta and do not translate directly to current
patch balance.

Data quality checks confirmed zero null values across the dataset. A
binomial test on side assignment confirmed team fairness. Average game
duration is 30.5 minutes. The dataset tracks 138 champions and contains
end-of-game totals only, with no mid-game state and no player skill
information such as rank or mastery.
