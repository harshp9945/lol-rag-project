# Champion Balance Findings

Champion balance was tested with 138 simultaneous binomial tests, one per
champion, asking whether each win rate deviates significantly from 50%.
Because 138 tests inflate false positives, the Benjamini-Hochberg
procedure was applied to control the false discovery rate across the
whole family of tests.

After correction, 46 of the 138 champions have win rates that differ
significantly from 50%: 23 are statistically overpowered (above 50%) and
23 are underpowered (below 50%).

The strongest overperformers, with 95% confidence intervals that sit
clearly above 50%, are Janna at 55.5% (over 8,600 games), Sona at 54.2%,
Yorick at 54.0%, Rammus at 53.9%, and Anivia at 53.6%. Because Janna is
both the highest win rate and one of the most contested champions (41.5%
ban rate), it is the clearest single balance outlier.

A Spearman correlation between ban rate and win rate across all champions
is weak but positive and significant (rho = 0.19, p = 0.02), meaning the
community's bans roughly align with which champions actually win, though
the relationship is loose.
