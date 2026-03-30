import pandas as pd
import numpy as np


def _add_zscore_per_query(df, column, eps=1e-6):
    """Compute z-score per query with stability guard."""
    grouped = df.groupby("qid")[column]
    std = grouped.transform("std").fillna(0.0)
    std = std.mask(std < eps, eps)
    mean = grouped.transform("mean").fillna(0.0)
    return ((df[column] - mean) / std).fillna(0.0)


def build_dataset(
    pair_identifiers,
    scores_A,
    scores_B,
    labels_df,
    retrieval_df,
    output_path: str = "train_ltr_pairs.parquet",
):
    """
    Builds the LTR dataset for XGBoost, including fused CE scores,
    hybrid scores, and robustness features.

    Args:
        pair_identifiers: List of (qid, pid) tuples aligned with scores_A/scores_B.
        scores_A: Scores from CrossEncoder A (e.g., BGE-reranker).
        scores_B: Scores from CrossEncoder B (e.g., MXBAI-reranker).
        labels_df: DataFrame containing at least ['qid', 'pid', 'label'] (original train/val).
        retrieval_df: DataFrame with retrieval scores for each candidate (qid, pid).
            Must contain: ['qid', 'pid', 'bm25_score', 'bge_bi_score'].
        output_path: Path to save the output .parquet file.

    Returns:
        train_ltr_df: Final DataFrame containing base features and labels.
        Also saved to disk as a parquet file.
    """

    # Base DataFrame with (qid, pid) and Cross-Encoder scores
    pairs_df = pd.DataFrame(pair_identifiers, columns=["qid", "pid"])
    pairs_df["score_bge_ce"] = scores_A
    pairs_df["score_mxbai_ce"] = scores_B

    # Add RRF rank (order of appearance in pair_identifiers per qid)
    pairs_df["rank_rrf"] = (
        pairs_df.groupby("qid").cumcount() + 1
    ).astype("int32")

    # Per-query normalization
    pairs_df["score_bge_ce_z"] = _add_zscore_per_query(pairs_df, "score_bge_ce")
    pairs_df["score_mxbai_ce_z"] = _add_zscore_per_query(pairs_df, "score_mxbai_ce")

    # Merge with retrieval scores (BM25/BGE)
    # Ensure retrieval_df contains required columns
    required_retrieval_cols = {"qid", "pid", "bm25_score", "bge_bi_score"}
    missing_retrieval = required_retrieval_cols - set(retrieval_df.columns)
    if missing_retrieval:
        raise ValueError(
            f"retrieval_df must contain columns {required_retrieval_cols}, "
            f"missing: {missing_retrieval}"
        )

    pairs_df = pairs_df.merge(
        retrieval_df,
        on=["qid", "pid"],
        how="left",
        validate="many_to_one",  # One row per (qid, pid) in retrieval_df
    )
    pairs_df[["bm25_score", "bge_bi_score"]] = pairs_df[["bm25_score", "bge_bi_score"]].fillna(0.0)

    # Retrieval normalization
    pairs_df["bm25_z"] = _add_zscore_per_query(pairs_df, "bm25_score")
    pairs_df["bge_bi_z"] = _add_zscore_per_query(pairs_df, "bge_bi_score")

    # Fused CE and hybrid score
    W_BGE = 0.5
    W_MXBAI = 0.5
    pairs_df["score_ce_fused"] = (
        W_BGE * pairs_df["score_bge_ce_z"] + W_MXBAI * pairs_df["score_mxbai_ce_z"]
    )

    W_CE = 0.6
    W_BM25 = 0.2
    W_BGE_BI = 0.2
    pairs_df["hybrid_score"] = (
        W_CE * pairs_df["score_ce_fused"]
        + W_BM25 * pairs_df["bm25_z"]
        + W_BGE_BI * pairs_df["bge_bi_z"]
    )

    # Robustness features
    diff_ce = pairs_df["score_bge_ce"] - pairs_df["score_mxbai_ce"]
    pairs_df["ce_var"] = (diff_ce ** 2) / 2.0
    pairs_df["ce_gap_abs"] = diff_ce.abs()
    pairs_df["ce_bm25_gap"] = pairs_df["score_ce_fused"] - pairs_df["bm25_z"]
    pairs_df["ce_bge_gap"] = pairs_df["score_ce_fused"] - pairs_df["bge_bi_z"]
    pairs_df["ce_conflict_flag"] = (
        (
            pairs_df["score_bge_ce_z"].fillna(0.0).apply(np.sign)
            != pairs_df["score_mxbai_ce_z"].fillna(0.0).apply(np.sign)
        )
        & (pairs_df["ce_gap_abs"] > 1.0)
    ).astype("int32")

    # Merge with original labels
    if not {"qid", "pid", "label"}.issubset(labels_df.columns):
        raise ValueError("labels_df must contain at least columns ['qid', 'pid', 'label'].")

    labels_sub = labels_df[["qid", "pid", "label"]].copy()
    train_ltr_df = pairs_df.merge(labels_sub, on=["qid", "pid"], how="left")
    train_ltr_df["label"] = train_ltr_df["label"].fillna(0).astype("int32")

    # Save to parquet
    train_ltr_df.to_parquet(output_path, index=False)
    print(f"Saved {output_path} with shape: {train_ltr_df.shape}")

    return train_ltr_df
