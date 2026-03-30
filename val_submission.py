import pandas as pd
import numpy as np
import string
import re
import pickle
import itertools
TAG_RE = re.compile(r"<[^>]+>")
WS_RE  = re.compile(r"\s+")

import torch as th

from tqdm import tqdm

# nltk imports
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from rank_bm25 import BM25Okapi

from sentence_transformers import SentenceTransformer, util, CrossEncoder
from xgboost import XGBRanker

from train_xgboost import add_extra_features

from build_dataset import build_dataset

from collections import defaultdict

nltk.download('stopwords')
device = 'cuda' if th.cuda.is_available() else 'cpu'
print(f'Working on {device}')

language = 'english'

# Stemming object 
stemmer = SnowballStemmer(language) # Words truncation
stop_words = set(stopwords.words(language))
translator = str.maketrans({c: " " for c in string.punctuation}) #  Tool for remove punctuation

def _add_zscore_per_query(df, column, eps=1e-6):
    """Compute z-score per query with small epsilon for stability."""
    grouped = df.groupby("qid")[column]
    std = grouped.transform("std").fillna(0.0)
    std = std.mask(std < eps, eps)
    mean = grouped.transform("mean").fillna(0.0)
    return ((df[column] - mean) / std).fillna(0.0)

def bm25_tokenizer(txt):
    """
    This methods make text cleaning and text tokening 
    """
    # Lower case & removeing punctuation
    txt = txt.lower().translate(translator)
    tokens = txt.split()

    # Remove stop-words and Stemming
    return [
        stemmer.stem(t)
        for t in tokens
        if t not in stop_words
    ]
    
def cross_clean(txt):
    """
    Remove special char
    """
    txt = TAG_RE.sub(" ", txt)
    txt = WS_RE.sub(" ", txt).strip()
    return txt

def execute_bm25(tokenized_corpus, map_queries, all_pids, unique_queries, top_k=150):
    """
    Execute Retrieval by BM25 for each query.
    Ritorna:
        - ranked_pids: dict[qid] -> [pid1, pid2, ...] (solo ranking, come prima)
        - score_dict:  dict[qid] -> {pid: score_bm25}
    """
    print("bm25 retrieval start")

    ranked_pids = {}
    score_dict = {}
    bm25 = BM25Okapi(tokenized_corpus)

    for qid in tqdm(unique_queries):
        query_text = map_queries[qid]
        scores = bm25.get_scores(bm25_tokenizer(query_text))  # vector di score per tutti i documenti

        # indici dei top-k ordinati per score decrescente
        top_n_indices = np.argsort(scores)[::-1][:top_k]
        pids = [all_pids[i] for i in top_n_indices]

        ranked_pids[qid] = pids
        # mappa pid -> score grezzo BM25
        score_dict[qid] = {
            pid: float(scores[idx]) for pid, idx in zip(pids, top_n_indices)
        }

    return ranked_pids, score_dict


def execute_bge(all_passage_list, map_queries, unique_qids, device="cpu", top_k=150):
    """
    Execute Retrieval by BGE for each query.
    Usa SentenceTransformer + semantic_search.
    Ritorna:
        - ranked_pids: dict[qid] -> [pid1, pid2, ...] (ranking)
        - score_dict:  dict[qid] -> {pid: score_bge}
    """
    print("bge retrieval start")

    ranked_pids = {}
    score_dict = {}

    bge = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)

    # Encoding corpus
    print("Encoding document ...")
    doc_embeddings = bge.encode(
        all_passage_list,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # Encoding query
    instruction = "Represent this sentence for searching relevant passages: "
    q_text = [instruction + map_queries[qid] for qid in unique_qids]
    q_embeddings = bge.encode(
        q_text,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # semantic_search restituisce lista per query di dict con 'corpus_id' e 'score'
    hits_per_query = util.semantic_search(q_embeddings, doc_embeddings, top_k=top_k)

    for i, hit_list in enumerate(hits_per_query):
        qid = unique_qids[i]
        pids = []
        pid2score = {}

        for h in hit_list:
            pid = all_pids[h["corpus_id"]]  # all_pids è globale, come già usato
            score = float(h["score"])
            pids.append(pid)
            pid2score[pid] = score

        ranked_pids[qid] = pids
        score_dict[qid] = pid2score

    del bge, doc_embeddings, q_embeddings
    th.cuda.empty_cache()

    return ranked_pids, score_dict


def rrf_fusion(run_a, run_b, k_rrf=10, top_k=200, w_a=1.0, w_b=1.0):
    fused = {}
    qids = sorted(set(run_a.keys()) | set(run_b.keys()))

    for qid in qids:
        la = run_a.get(qid, [])
        lb = run_b.get(qid, [])

        rank_a = {pid: r for r, pid in enumerate(la, start=1)}
        rank_b = {pid: r for r, pid in enumerate(lb, start=1)}

        # pool = unione, ma con ranking fuso
        pool = set(rank_a.keys()) | set(rank_b.keys())

        scored = []
        for pid in pool:
            ra = rank_a.get(pid)
            rb = rank_b.get(pid)
            score = 0.0
            if ra is not None:
                score += w_a * (1.0 / (k_rrf + ra))
            if rb is not None:
                score += w_b * (1.0 / (k_rrf + rb))
            scored.append((pid, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        fused[qid] = [pid for pid, _ in scored[:top_k]]

    return fused

def compute_ap(ranked_list, relevant_set, k = 10):
    """
    Compute Average Precision @ 10 for a query
    """
    hits = 0
    s = 0.0
    for i, pid in enumerate(ranked_list[:k], start=1):
        if pid in relevant_set:
            hits += 1
            s += hits / i
    denom = min(len(relevant_set), k)
    return s / denom if denom > 0 else 0.0

def compute_map(predictions, ground_truth, qids, k = 10):
    """
    Compute Mean Avarage Precision @ 10 for a query
    """
    ap_scores = []
    for qid in qids:
        rel = ground_truth.get(qid, set())
        if len(rel) == 0:
            continue
        ranked = predictions.get(qid, [])
        ap_scores.append(compute_ap(ranked, rel, k=k))
    return float(np.mean(ap_scores)) if ap_scores else 0.0

def get_model_scores(model_name, all_pairs, batch_size, device='cpu'):
    print(f'Loading model {model_name}')

    # Robust path for some v2 rerankers (sequence classification)
    if "mxbai-rerank-" in model_name and model_name.endswith("-v2"):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True
        ).to(device)
        model.eval()

        scores = []
        for i in range(0, len(all_pairs), batch_size):
            batch = all_pairs[i:i + batch_size]
            q = [x[0] for x in batch]
            d = [x[1] for x in batch]

            enc = tok(
                q, d,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)

            with th.no_grad():
                out = model(**enc)
                logits = out.logits
                if logits.shape[-1] == 1:
                    s = logits.squeeze(-1)
                else:
                    s = logits[:, 1]
                scores.append(s.detach().float().cpu().numpy())

        del model
        th.cuda.empty_cache()
        return np.concatenate(scores, axis=0)

    # Default: sentence-transformers CrossEncoder
    model = CrossEncoder(model_name, device=device, max_length=512)
    scores = model.predict(all_pairs, batch_size=batch_size, show_progress_bar=True)
    del model
    th.cuda.empty_cache()
    return scores

# Read validation set
validation_url = 'Data/val.parquet'
val_set = pd.read_parquet(validation_url)

# Read collections set
collection_url = 'Data/collection.parquet'
document_set = pd.read_parquet(collection_url)

# Setting data structure
map_queries = val_set.groupby('qid')['query'].first().to_dict()
map_passages = dict(zip(document_set['pid'], document_set['passage']))

unique_qids = list(map_queries.keys())
all_pids = list(map_passages.keys())
all_passages_list = list(map_passages.values())

# Get ground truth
ground_truth = (
    val_set[val_set["label"] == 1]
    .groupby("qid")["pid"]
    .apply(lambda s: set(s.tolist()))
    .to_dict()
)
for qid in unique_qids:
    if qid not in ground_truth:
        ground_truth[qid] = set()

# ---- BM25 and BGE Retrieval ---- 
print('Tokenizing corpus')
tokenized_corpus = [bm25_tokenizer(doc) for doc in tqdm(all_passages_list)]

bm25_ranking, bm25_raw_scores = execute_bm25(tokenized_corpus, map_queries, all_pids, unique_qids)
bge_ranking, bge_raw_scores = execute_bge(all_passages_list, map_queries, unique_qids, device)

# Merge BGE and BM25 scores
candidates = rrf_fusion(bm25_ranking, bge_ranking)
with open("candidates.pkl", "wb") as f:
    pickle.dump(candidates, f, protocol=pickle.HIGHEST_PROTOCOL)

# Build data frame with BGE and BM25 scores
rows = []
for qid in unique_qids:
    bm = bm25_raw_scores.get(qid, {})
    bg = bge_raw_scores.get(qid, {})
    pool_pids = set(bm.keys()) | set(bg.keys())
    for pid in pool_pids:
        rows.append(
            {
                "qid": qid,
                "pid": pid,
                "bm25_score": bm.get(pid, 0.0),
                "bge_bi_score": bg.get(pid, 0.0),
            }
        )
retrieval_df = pd.DataFrame(rows)
# -------------
retrieval_candidates = defaultdict(set)
for _, row in retrieval_df.iterrows():
    retrieval_candidates[row["qid"]].add(row["pid"])

num_q = 0
num_with_pos = 0
for qid in unique_qids:
    rel = ground_truth.get(qid, set())
    if len(rel) == 0:
        continue  # niente rilevanti in ground truth -> non contiamo questa query
    num_q += 1
    # intersezione tra candidati e rilevanti
    if len(rel & retrieval_candidates.get(qid, set())) > 0:
        num_with_pos += 1

recall_queries = num_with_pos / num_q if num_q > 0 else 0.0
print(f"Recall@pool (almeno 1 rilevante nel pool): {recall_queries:.3f}")

# Cross Encoders pre-elaboration
flat_pairs = []
pair_identifiers = []

print('Preparing Pairs')
for qid, pids in tqdm(candidates.items()):
    q_txt = map_queries[qid]
    for pid in pids:
        p_txt = cross_clean(map_passages.get(pid, ''))
        flat_pairs.append([q_txt, p_txt])
        pair_identifiers.append((qid, pid))


# Setup cross Encoders
model_a = 'BAAI/bge-reranker-v2-m3'  
model_b = 'mixedbread-ai/mxbai-rerank-base-v1'
batch_size = 128

# Score all pairs with cross-encoders 
scores_A = get_model_scores(model_a, flat_pairs, batch_size, device)
scores_B = get_model_scores(model_b, flat_pairs, batch_size, device)

# build_dataset(
#     pair_identifiers=pair_identifiers,
#     scores_A = scores_A,
#     scores_B = scores_B,
#     retrieval_df=retrieval_df,
#     labels_df = val_set,
# )

# Build DF with same training features
ltr_infer = pd.DataFrame(pair_identifiers, columns=["qid", "pid"])
ltr_infer["score_bge_ce"]   = scores_A
ltr_infer["score_mxbai_ce"] = scores_B
ltr_infer["rank_rrf"]       = ltr_infer.groupby("qid").cumcount() + 1

ltr_infer = ltr_infer.merge(
    retrieval_df[["qid", "pid", "bm25_score", "bge_bi_score"]],
    on=["qid", "pid"],
    how="left",
)
ltr_infer[["bm25_score", "bge_bi_score"]] = ltr_infer[["bm25_score", "bge_bi_score"]].fillna(0.0)

# Per-query normalization and fusion
ltr_infer["score_bge_ce_z"] = _add_zscore_per_query(ltr_infer, "score_bge_ce")
ltr_infer["score_mxbai_ce_z"] = _add_zscore_per_query(ltr_infer, "score_mxbai_ce")
ltr_infer["bm25_z"] = _add_zscore_per_query(ltr_infer, "bm25_score")
ltr_infer["bge_bi_z"] = _add_zscore_per_query(ltr_infer, "bge_bi_score")

# Robustness features to catch disagreement between scorers
diff_ce = ltr_infer["score_bge_ce"] - ltr_infer["score_mxbai_ce"]
ltr_infer["ce_var"] = (diff_ce ** 2) / 2.0
ltr_infer["ce_gap_abs"] = diff_ce.abs()
ltr_infer["ce_bm25_gap"] = ltr_infer["score_ce_fused"] - ltr_infer["bm25_z"]
ltr_infer["ce_bge_gap"] = ltr_infer["score_ce_fused"] - ltr_infer["bge_bi_z"]
ltr_infer["ce_conflict_flag"] = (
    (
        ltr_infer["score_bge_ce_z"].fillna(0.0).apply(np.sign)
        != ltr_infer["score_mxbai_ce_z"].fillna(0.0).apply(np.sign)
    )
    & (ltr_infer["ce_gap_abs"] > 1.0)
).astype("int32")

ltr_feature = add_extra_features(ltr_infer, map_queries, map_passages)
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

param_grid = {
    'W_BGE': [0.4, 0.5, 0.6],       # Peso per BGE-CrossEncoder
    # W_MXBAI sarà derivato (o puoi metterlo in griglia se vuoi indipendenza)
    'W_CE': [0.5, 0.6, 0.7],        # Peso globale CrossEncoder in hybrid_score
    'W_BM25': [0.1, 0.2, 0.3],      # Peso BM25 in hybrid_score
    'W_BGE_BI': [0.1, 0.2, 0.3]     # Peso BGE Bi-Encoder in hybrid_score
}

keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Starting Grid Search with {len(combinations)} combinations...")

best_map = -1.0
best_params = {}
best_preds = {}

ranker = XGBRanker()
ranker.load_model("xgb_ltr_best.json")

for i, params in enumerate(tqdm(combinations)):
    w_bge = params['W_BGE']
    w_mxbai = 1.0 - w_bge

    w_ce = params['W_CE']
    w_bm25 = params['W_BM25']
    w_bge_bi = params['W_BGE_BI']

    ltr_infer["score_ce_fused"] = (
        w_bge * ltr_infer["score_bge_ce_z"] + w_mxbai * ltr_infer["score_mxbai_ce_z"]
    )

    ltr_infer["hybrid_score"] = (
        w_ce * ltr_infer["score_ce_fused"]
        + w_bm25 * ltr_infer["bm25_z"]
        + w_bge_bi * ltr_infer["bge_bi_z"]
    )
    
    # Calcolo gap features che dipendono da score_ce_fused
    ltr_infer["ce_bm25_gap"] = ltr_infer["score_ce_fused"] - ltr_infer["bm25_z"]
    ltr_infer["ce_bge_gap"] = ltr_infer["score_ce_fused"] - ltr_infer["bge_bi_z"]

    current_preds = {}

    X_all = ltr_infer[feature_cols].values.astype("float32")
    all_scores = ranker.predict(X_all)

    ltr_infer["_temp_xgb_score"] = all_scores

    top_k_df = ltr_infer.sort_values(["qid", "_temp_xgb_score"], ascending=[True, False]).groupby("qid").head(10)

    current_preds = top_k_df.groupby("qid")["pid"].apply(list).to_dict()

    current_map = compute_map(current_preds, ground_truth, unique_qids)

    if current_map > best_map:
        best_map = current_map
        best_params = params.copy()
        best_params['W_MXBAI'] = w_mxbai # Salviamo anche questo per chiarezza
        best_preds = current_preds

print("-" * 30)
print(f"GRID SEARCH COMPLETED")
print(f"Best MAP@10: {best_map:.5f}")
print(f"Best Parameters: {best_params}")
print("-" * 30)