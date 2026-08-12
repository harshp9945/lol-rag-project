"""
build_knowledge_base.py
------------------------
Converts the statistical findings from the LoL Match Analytics capstone
into structured, retrievable documents for the RAG system.

This is the most important file in the project conceptually: it defines
WHAT the assistant actually knows. Each document is a self-contained,
factually grounded statement with its source notebook cited — this is
what prevents hallucination later. The LLM can only answer from these
documents, not from its own training data about League of Legends.
"""

import json
from pathlib import Path

DOCS_DIR = Path(__file__).parent.parent / "data" / "knowledge_base"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Each entry is a grounded fact pulled directly from the capstone's
# actual statistical output. In a real build, these would be generated
# programmatically from the notebook outputs — here they're hand-curated
# from the real numbers we computed earlier in the LoL project.

KNOWLEDGE_DOCUMENTS = [
    {
        "id": "obj_001",
        "source_notebook": "02_match_outcome_drivers.ipynb",
        "topic": "objective_win_rates",
        "content": (
            "First Inhibitor win rate: The team that secures the first inhibitor "
            "wins 91.1% of matches. This is the single strongest objective predictor "
            "in the dataset (p<0.001, large effect size). 95% confidence interval: "
            "90.4% to 91.8%, based on 51,490 ranked matches from Season 9."
        ),
    },
    {
        "id": "obj_002",
        "source_notebook": "02_match_outcome_drivers.ipynb",
        "topic": "objective_win_rates",
        "content": (
            "First Baron win rate: The team that secures the first Baron Nashor "
            "wins 81.2% of matches (p<0.001, medium-large effect size, Cramer's V "
            "approximately 0.29). This is the second-strongest single objective "
            "predictor after First Inhibitor."
        ),
    },
    {
        "id": "obj_003",
        "source_notebook": "02_match_outcome_drivers.ipynb",
        "topic": "objective_win_rates",
        "content": (
            "First Tower win rate: The team that destroys the first tower wins "
            "70.8% of matches (p<0.001, medium effect size). This reflects early "
            "structural advantage translating into a measurable, but not decisive, "
            "win probability increase."
        ),
    },
    {
        "id": "obj_004",
        "source_notebook": "02_match_outcome_drivers.ipynb",
        "topic": "objective_win_rates",
        "content": (
            "First Dragon win rate: The team that secures the first dragon wins "
            "68.0% of matches. First Blood (first kill of the match) has the weakest "
            "effect of all early objectives at 59.5% win rate, statistically "
            "significant but practically the smallest effect size of the six "
            "first-objectives tested."
        ),
    },
    {
        "id": "combo_001",
        "source_notebook": "09_objective_combinations_and_strategy.ipynb",
        "topic": "objective_combinations",
        "content": (
            "When a team secures Baron, Dragon, and Tower objectives together, the "
            "combined win rate rises to 89.6%. This is higher than any single "
            "objective alone except First Inhibitor, suggesting these three "
            "objectives compound rather than simply add independently."
        ),
    },
    {
        "id": "combo_002",
        "source_notebook": "09_objective_combinations_and_strategy.ipynb",
        "topic": "interaction_effects",
        "content": (
            "Logistic regression with interaction terms shows the Baron x Dragon "
            "interaction has a positive coefficient, indicating these two objectives "
            "are synergistic rather than redundant: securing both produces a larger "
            "win probability increase than the sum of securing each independently."
        ),
    },
    {
        "id": "balance_001",
        "source_notebook": "03_champion_balance_analysis.ipynb",
        "topic": "champion_balance",
        "content": (
            "Champion balance was tested using binomial tests across 138 champions "
            "with Benjamini-Hochberg false discovery rate correction to account for "
            "multiple simultaneous testing. After correction, a subset of champions "
            "show statistically significant win rates above 53% (overpowered "
            "candidates) and below 47% (underpowered candidates) at the 95% "
            "confidence level."
        ),
    },
    {
        "id": "balance_002",
        "source_notebook": "11_ban_and_pick_meta_analysis.ipynb",
        "topic": "ban_rate_meta",
        "content": (
            "Yasuo has a ban rate of approximately 64%, the highest in the dataset, "
            "despite not having the single highest win rate among all 138 champions. "
            "This indicates the community's perceived threat level (reflected in ban "
            "rate) does not perfectly correlate with actual statistical win rate, "
            "based on Spearman rank correlation analysis between ban rate and win rate."
        ),
    },
    {
        "id": "balance_003",
        "source_notebook": "11_ban_and_pick_meta_analysis.ipynb",
        "topic": "stealth_op_champions",
        "content": (
            "'Stealth OP' champions are defined as those with statistically "
            "significant win rates above 53% but ban rates below 10%, indicating "
            "the player community is underestimating their actual strength. These "
            "champions represent the highest-value targets for proactive balance "
            "review before they become widely discovered."
        ),
    },
    {
        "id": "duration_001",
        "source_notebook": "04_match_duration_and_game_health.ipynb",
        "topic": "game_duration",
        "content": (
            "Average game duration in the dataset is approximately 30.5 minutes. "
            "A Kolmogorov-Smirnov test confirms game duration is NOT normally "
            "distributed (right-skewed), which is why non-parametric tests like "
            "Mann-Whitney U were used instead of t-tests when comparing duration "
            "groups."
        ),
    },
    {
        "id": "snowball_001",
        "source_notebook": "05_snowball_effect_analysis.ipynb",
        "topic": "comeback_rate",
        "content": (
            "Teams that are behind in tower kills still win approximately 27% of "
            "matches, indicating comebacks are possible but not common. Teams ahead "
            "in tower kills win roughly 70-75% of the time. This suggests the game "
            "rewards early leads without making them fully deterministic."
        ),
    },
    {
        "id": "model_001",
        "source_notebook": "06_win_prediction_model.ipynb",
        "topic": "win_prediction_model",
        "content": (
            "A HistGradientBoosting model trained on 33 engineered features achieves "
            "an AUC-ROC of 0.9977 using 5-fold stratified cross-validation. However, "
            "a model using only 3 features (First Inhibitor, First Baron, First "
            "Tower) already achieves 0.90 AUC. This demonstrates that late-game "
            "objective control is near-deterministic for match outcome -- the high "
            "AUC reflects the dataset's nature (post-game objectives correlate "
            "strongly with the outcome they helped cause) rather than exceptional "
            "model sophistication."
        ),
    },
    {
        "id": "model_002",
        "source_notebook": "06_win_prediction_model.ipynb",
        "topic": "feature_importance",
        "content": (
            "Permutation feature importance on the win prediction model shows tower "
            "kill advantage as the single most important feature, followed by "
            "inhibitor-related features. First Blood and First Rift Herald have the "
            "lowest permutation importance among all 33 features tested, consistent "
            "with their weaker standalone win rate effects."
        ),
    },
    {
        "id": "synergy_001",
        "source_notebook": "10_team_composition_analysis.ipynb",
        "topic": "champion_synergy",
        "content": (
            "Champion synergy was analyzed for the top 20 champions by pick rate, "
            "examining win rates when pairs of champions were on the same team "
            "(minimum 50 games per pair for statistical reliability). Some pairs "
            "show win rates significantly above 50%, suggesting positive kit "
            "synergy beyond what either champion achieves individually."
        ),
    },
]


def build_knowledge_base():
    """Write each knowledge document as a separate JSON file for easy retrieval indexing."""
    manifest = []
    for doc in KNOWLEDGE_DOCUMENTS:
        filepath = DOCS_DIR / f"{doc['id']}.json"
        with open(filepath, "w") as f:
            json.dump(doc, f, indent=2)
        manifest.append(doc["id"])

    # Write a manifest for quick reference
    with open(DOCS_DIR / "_manifest.json", "w") as f:
        json.dump({"total_documents": len(manifest), "document_ids": manifest}, f, indent=2)

    print(f"Knowledge base built: {len(KNOWLEDGE_DOCUMENTS)} documents written to {DOCS_DIR}")
    return KNOWLEDGE_DOCUMENTS


if __name__ == "__main__":
    build_knowledge_base()
