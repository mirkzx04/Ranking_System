import pandas as pd
import numpy as np
import re
import string

from collections import defaultdict

from sklearn.model_selection import GroupShuffleSplit, ParameterSampler
from xgboost import XGBRanker
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from tqdm import tqdm


# Base dataset already built for XGBoost:
# Must contain at least: qid, pid, label, score_bge_ce, score_mxbai_ce, rank_rrf
LTR_BASE_PATH    = "train_ltr_pairs.parquet"

# Query file 
TRAIN_QUERIES    = "Data/train.parquet" 

# Complete collection with document texts
COLLECTION_PATH  = "Data/collection.parquet"

# Where to save the best XGBoost model
MODEL_OUT_PATH   = "xgb_ltr_best.json"

LANGUAGE = "english"
stemmer = SnowballStemmer(LANGUAGE)
stop_words = set(stopwords.words(LANGUAGE))
translator = str.maketrans({c: " " for c in string.punctuation})

digit_pattern = re.compile(r"\d+")

def bm25_style_tokenize(text: str):
    txt = text.lower().translate(translator)
    tokens = txt.split()
    return [stemmer.stem(t) for t in tokens if t not in stop_words]

def lexical_features(q_text: str, d_text: str):
    q_tokens = bm25_style_tokenize(q_text)
    d_tokens = bm25_style_tokenize(d_text)

    q_set = set(q_tokens)
    d_set = set(d_tokens)

    overlap = len(q_set & d_set)
    q_len = len(q_tokens)
    d_len = len(d_tokens)

    q_digits = set(digit_pattern.findall(q_text))
    has_all_digits = int(len(q_digits) > 0 and all(d in d_text for d in q_digits))

    return overlap, q_len, d_len, has_all_digits

def add_extra_features(
    base_df: pd.DataFrame,
    map_queries: dict,
    map_passages: dict,
):
    """
    Adds lexical features and rank normalization.
    Requires base_df to contain at least:
      qid, pid, score_bge_ce, score_mxbai_ce, rank_rrf
    """
    term_overlap = []
    query_len    = []
    doc_len      = []
    digits_match = []

    for qid, pid in tqdm(
        zip(base_df["qid"], base_df["pid"]),
        total=len(base_df),
        desc="Calculating lexical features",
    ):
        q_text = map_queries.get(qid, "")
        d_text = map_passages.get(pid, "")
        ov, ql, dl, dm = lexical_features(q_text, d_text)
        term_overlap.append(ov)
        query_len.append(ql)
        doc_len.append(dl)
        digits_match.append(dm)

    base_df["term_overlap"] = term_overlap
    base_df["query_len"]    = query_len
    base_df["doc_len"]      = doc_len
    base_df["digits_match"] = digits_match

    base_df["rank_rrf_invlog"] = 1.0 / np.log1p(base_df["rank_rrf"] + 1)

    return base_df

def group_from_qids(qid_array: np.ndarray):
    _, counts = np.unique(qid_array, return_counts=True)
    return counts.tolist()

def map_at_k(qids, labels, scores, k=10):
    data = defaultdict(list)
    for qid, y, s in zip(qids, labels, scores):
        data[qid].append((s, y))

    ap_list = []
    for lst in data.values():
        lst.sort(key=lambda x: x[0], reverse=True)
        topk = lst[:k]
        total_rel = sum(y for _, y in lst)
        if total_rel == 0:
            ap_list.append(0.0)
            continue
        num_rel = 0
        precisions = []
        for rank, (_, y) in enumerate(topk, start=1):
            if y > 0:
                num_rel += 1
                precisions.append(num_rel / rank)
        ap_list.append(sum(precisions) / min(k, total_rel))
    return float(np.mean(ap_list)) if ap_list else 0.0

def main():
    print("Loading base LTR dataset...")
    df = pd.read_parquet(LTR_BASE_PATH)
    print("Initial Shape:", df.shape)

    required_cols = {
        "qid",
        "pid",
        "label",
        "score_bge_ce",
        "score_mxbai_ce",
        "score_bge_ce_z",
        "score_mxbai_ce_z",
        "score_ce_fused",
        "hybrid_score",
        "bm25_score",
        "bge_bi_score",
        "bm25_z",
        "bge_bi_z",
        "rank_rrf",
        "ce_var",
        "ce_gap_abs",
        "ce_bm25_gap",
        "ce_bge_gap",
        "ce_conflict_flag",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in base dataset: {missing}")
    
    print("Loading train queries and collection for lexical features...")
    train_df = pd.read_parquet(TRAIN_QUERIES)         # "Data/train.parquet"
    coll_df  = pd.read_parquet(COLLECTION_PATH) 

    map_queries  = train_df.groupby("qid")["query"].first().to_dict()
    map_passages = dict(zip(coll_df["pid"], coll_df["passage"]))

    df = add_extra_features(df, map_queries, map_passages)

    # FEATURES USED BY XGBOOST 
    feature_cols = [
        "score_bge_ce",
        "score_mxbai_ce",
        "score_bge_ce_z",
        "score_mxbai_ce_z",
        "score_ce_fused",
        "hybrid_score",
        "bm25_score",
        "bge_bi_score",
        "bm25_z",
        "bge_bi_z",
        "rank_rrf",
        "rank_rrf_invlog",
        "term_overlap",
        "query_len",
        "doc_len",
        "digits_match",
        "ce_var",
        "ce_gap_abs",
        "ce_bm25_gap",
        "ce_bge_gap",
        "ce_conflict_flag",
    ]

    X = df[feature_cols].values.astype("float32")
    y = df["label"].values.astype("float32")
    qids = df["qid"].values

    print("Building train/val split by query...")
    gss = GroupShuffleSplit(test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups=qids))

    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    qid_tr, qid_val = qids[train_idx], qids[val_idx]

    group_tr  = group_from_qids(qid_tr)
    group_val = group_from_qids(qid_val)

    # HYPERPARAMETER RANDOM SEARCH
    base_params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@10",
        "random_state": 42,
        "tree_method": "hist",
        'device' : 'cuda'
    }

    param_distributions = {
        "max_depth":        [4, 5, 6, 7],
        "learning_rate":    [0.03, 0.05, 0.08],
        "n_estimators":     [300, 500, 800],
        "subsample":        [0.7, 0.85, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 5, 10],
        "reg_lambda":       [0.5, 1.0, 2.0, 3.0],
        "reg_alpha":        [0.0, 0.5, 1.0],
    }

    n_iter = 25
    sampler = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=42))

    best_score = -1.0
    best_params = None
    best_model  = None

    print(f"Starting random search on {n_iter} configurations...")
    for i, params in tqdm(enumerate(sampler, start=1), total=n_iter, desc="Random Search"):
        tqdm.write(f"[{i}/{n_iter}] Testing config: {params}")
        ranker = XGBRanker(**base_params, **params)
        ranker.fit(
            X_tr, y_tr,
            group=group_tr,
            eval_set=[(X_val, y_val)],
            eval_group=[group_val],
            verbose=False,
        )
        scores_val = ranker.predict(X_val)
        map10 = map_at_k(qid_val, y_val, scores_val, k=10)
        tqdm.write(f"    MAP@10 validation = {map10:.4f}")

        if map10 > best_score:
            best_score = map10
            best_params = params
            best_model  = ranker

    print("=== RANDOM SEARCH RESULTS ===")
    print("Best MAP@10:", best_score)
    print("Best parameters:", best_params)

    if best_model is not None:
        print(f"Saving the best model to {MODEL_OUT_PATH}")
        best_model.save_model(MODEL_OUT_PATH)

if __name__ == "__main__":
    main()
