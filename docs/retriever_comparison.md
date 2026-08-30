# Retriever Comparison: TF-IDF vs Embeddings

Both retrievers implement the same `Retriever` interface, so they were
swapped and measured on identical query sets using
`evals/calibrate_thresholds.py`.

## The question

The refusal guardrail needs a threshold that separates in-scope questions
(answerable from the capstone corpus) from out-of-scope ones (unrelated,
or gaming questions the corpus doesn't cover). A clean separation means a
single threshold can answer every good question and refuse every bad one.

## Results

| Retriever  | In-scope top-1 (min / median / max) | Out-of-scope top-1 (min / median / max) | Clean separation? |
|------------|-------------------------------------|-----------------------------------------|-------------------|
| TF-IDF     | 0.146 / — / —                       | — / — / 0.194                           | No (overlap)      |
| Embeddings | 0.448 / 0.615 / 0.736               | 0.046 / 0.332 / 0.400                   | Yes               |

With TF-IDF, the lowest in-scope score (0.146) fell *below* the highest
out-of-scope score (0.194): no single threshold works. A question like
"which champions are flagged as balance outliers" was refused, while a
gaming-vocabulary question like "best Yasuo build" leaked through, because
TF-IDF matches words and both share tokens with the corpus.

With embeddings (all-MiniLM-L6-v2 + Chroma, cosine), the lowest in-scope
score (0.448) sits clearly above the highest out-of-scope score (0.400).
The calibration script suggests a threshold of 0.424, which answers every
in-scope query and refuses every out-of-scope one on the test sets.

## Why

TF-IDF scores on shared vocabulary; gaming terms leak similarity from
unrelated questions. Embeddings score on meaning, so "balance outliers"
lands near "champions that deviate significantly from 50%" even with no
shared words, while an unrelated gaming question stays far away.

## Trade-offs

Embeddings are not free: they need a model download (~90MB), are slower to
index and query than TF-IDF, and pull in heavier dependencies (PyTorch via
sentence-transformers, plus Chroma). For this corpus the retrieval-quality
gain clearly justifies the cost. For a tiny corpus with only exact-keyword
queries, TF-IDF would remain a reasonable, dependency-light choice. Keeping
both behind one interface makes the trade-off explicit and reversible.