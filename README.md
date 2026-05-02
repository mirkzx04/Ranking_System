
```markdown
# Learning-to-Rank Pipeline for Passage Retrieval

This repository presents a multi-stage information retrieval and learning-to-rank pipeline developed for the Kaggle competition [AI/ML A.A. 2025-26 End of Year Hackathon](https://www.kaggle.com/competitions/aiml-a-a-2025-26-end-of-year-hackathon/leaderboard).

**Competition result:** 2nd place on the private leaderboard with a reported **MAP@10 = 0.71**  
**Kaggle team:** `AVENGERS`  
**Main components:** BM25, dense retrieval, neural reranking, XGBoost Learning-to-Rank

---

## Abstract

The goal of this project is to rank candidate passages for a query so that the most relevant documents appear in the top positions of the final list. The system combines lexical retrieval, dense retrieval, neural reranking, and a final supervised learning-to-rank model.

From a systems perspective, the pipeline is designed around a common information retrieval principle: use efficient methods to maximize recall in the first stage, then apply stronger but more expensive models on a much smaller candidate pool. The final ranking function is learned with gradient-boosted decision trees over engineered retrieval and reranking features.

---

## Competition context

The competition can be framed as a passage retrieval problem. Given a query $q$, the task is to identify and rank the most relevant passages $d \in \mathcal{D}$ from a large collection.

The final target metric is **MAP@10**, so the objective is not merely to classify passages as relevant or non-relevant, but to place the most relevant ones as high as possible in the top-10 ranking.

Formally, the system aims to learn a scoring function

$$
f_\theta(q,d) \in \mathbb{R}
$$

such that documents are sorted by decreasing $f_\theta(q,d)$, maximizing ranking quality at the top of the list.

---

## Method overview

The pipeline is divided into two main stages:

1. **Candidate generation (recall-oriented)**
   - BM25 is used for lexical retrieval.
   - A BGE bi-encoder is used for dense semantic retrieval.
   - The candidate lists are merged with Reciprocal Rank Fusion (RRF).

2. **Reranking and Learning-to-Rank (precision-oriented)**
   - Two cross-encoders compute fine-grained relevance scores.
   - A feature table is built from retrieval and reranking signals.
   - An `XGBRanker` learns the final ranking function.

---

## Training workflow
```text
      +-----------------+
      | Queries & Pairs |
      |       {d}       |
      +--------+--------+
               |
      +--------+--------+
      |                 |
      v                 v
    +----+         +--------+
    |BM25|         |BGE Dual|
    +--+-+         +---+----+
       |               |
       +---->+---+<----+
             |RRF|
             +-+-+
               |
               v
          +---------+
          |Top-K Set|
          |   {d}   |
          +----+----+
               |
     +---------+---------+
     |         |         |
     v         v         v
   +---+     +---+     +-----+
   |BGE|     |MXB|     |Base |
   |CE |     |CE |     |Feats|
   +-+-+     +-+-+     +--+--+
     |         |          |
     +---->+---+---+<-----+
           |Feature|
           |Eng.   |
           +---+---+
               |
               v
         +-----------+
         |LTR Dataset|
         |    {d}    |
         +-----+-----+
               |
               v
       +---------------+
       |train_xgboost  |
       +-------+-------+
               |
               v
         +-----------+
         | XGBRanker |
         +-----+-----+
               |
               v
        +-------------+
        |xgb_best.json|
        |     {s}     |
        +-------------+
```

The training phase transforms retrieval outputs and neural scores into a structured tabular dataset. The final `XGBRanker` is then trained to predict a ranking score for each query-document pair.

---

## Inference workflow
```text
            +-------+
            | Query |
            +---+---+
                |
       +--------+--------+
       |                 |
       v                 v
     +----+         +--------+
     |BM25|         |BGE Dual|
     +--+-+         +---+----+
        |               |
        +---->+---+<----+
              |RRF|
              +-+-+
                |
                v
           +---------+
           |Top-K Set|
           +----+----+
                |
      +---------+---------+
      |         |         |
      v         v         v
    +---+     +---+     +-----+
    |BGE|     |MXB|     |Base |
    |CE |     |CE |     |Feats|
    +-+-+     +-+-+     +--+--+
      |         |          |
      +---->+---+---+<-----+
            |Feature|
            |Constr.|
            +---+---+
                |          +-------------+
                v          |xgb_best.json|
          +-----------+    |     {s}     |
          | Inference |<---+-------------+
          +-----+-----+
                |
                v
          +-----------+
          |  Top-10   |
          |    {d}    |
          +-----------+
```

At inference time, the same feature logic used during training is applied to unseen queries. The trained XGBoost ranker combines all available signals and outputs the final ranking.

---

## Mathematical formulation

### Reciprocal Rank Fusion

If $r_m(d)$ denotes the rank assigned to document $d$ by retrieval model $m$, the RRF score is

$$
\mathrm{RRF}(d) = \sum_{m=1}^{M} \frac{1}{k + r_m(d)}
$$

where $k > 0$ is a damping constant and $M$ is the number of retrieval systems being fused.

### Per-query normalization

Because score distributions vary significantly across queries, several features are normalized at query level. Given a raw score $s(q,d)$, the normalized feature is

$$
z(q,d) = \frac{s(q,d) - \mu_q}{\sigma_q + \varepsilon}
$$

where $\mu_q$ and $\sigma_q$ are the mean and standard deviation for query $q$, and $\varepsilon$ is a small stability constant.

### Fused cross-encoder score

The two normalized cross-encoder outputs are combined through a simple linear fusion:

$$
s_{\mathrm{CE}}(q,d) = 0.5\, z_{\mathrm{BGE}}(q,d) + 0.5\, z_{\mathrm{MXBAI}}(q,d)
$$

### Hybrid score

The repository also defines a hybrid score that combines reranking and retrieval evidence:

$$
s_{\mathrm{hyb}}(q,d) = 0.6\, s_{\mathrm{CE}}(q,d) + 0.2\, z_{\mathrm{BM25}}(q,d) + 0.2\, z_{\mathrm{BI}}(q,d)
$$

This score acts as an interpretable aggregation of semantic reranking, lexical retrieval, and dense retrieval information.

---

## Feature engineering

A central part of the project is the construction of a learning-to-rank dataset from query-document pairs.

For each pair $(q,d)$, the pipeline includes:

- Raw cross-encoder scores.
- BM25 and dense retrieval scores.
- Per-query z-score normalized versions of these signals.
- A fused cross-encoder score.
- A hybrid score combining reranking and retrieval features.
- Robustness features describing model agreement and disagreement.

The feature engineering stage is particularly important because it allows the final ranker to use not only absolute relevance signals, but also uncertainty and conflict patterns across models.

Examples of robustness-oriented features include:

- Cross-encoder variance.
- Absolute gap between cross-encoder scores.
- Gap between fused neural score and BM25 score.
- Gap between fused neural score and dense retrieval score.
- A conflict flag identifying strong disagreement between normalized reranker outputs.

---

## Repository structure

| File | Role |
|------|------|
| `build_dataset.py` | Builds the learning-to-rank dataset from retrieval outputs, reranker scores, and labels |
| `train_xgboost.py` | Trains the final `XGBRanker` model |
| `val_submission.py` | Evaluates the full pipeline and computes MAP@10 on validation data |
| `xgb_ltr_best.json` | Stores the best trained XGBoost ranking model |
| `pyproject.toml` | Project configuration and dependencies |
| `uv.lock` | Locked dependency versions for reproducibility |

---

## Execution flow

A typical workflow is:

1. Prepare the retrieval outputs and labeled query-document pairs.
2. Build the tabular ranking dataset.
3. Train the XGBoost learning-to-rank model.
4. Evaluate the full ranking pipeline.

Example commands:
```bash
python build_dataset.py
python train_xgboost.py
python val_submission.py
```

---

## Why this approach works

The pipeline separates **recall optimization** from **precision optimization**. This is computationally efficient because expensive cross-encoders are applied only to a restricted candidate pool rather than to the full corpus.

The final learning-to-rank stage is useful because it does not rely on a single score. Instead, it learns how to combine lexical evidence, dense similarity, neural reranking outputs, and disagreement features into a single ranking function.

---

## Results

According to the current project report, the system achieved **2nd place** on the competition private leaderboard with a **MAP@10 score of 0.71**.

Beyond the leaderboard result, the repository can be viewed as a compact example of a modern retrieval architecture combining:

- lexical retrieval,
- dense retrieval,
- neural reranking,
- feature engineering,
- and gradient-boosted learning-to-rank.

---
```