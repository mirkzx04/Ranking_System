# Information Retrieval Pipeline with Learning-to-Rank (LTR)

🏆 **Kaggle Hackathon - 2nd Place Solution** 🏆

This repository contains our team's solution for the [AI/ML A.A. 2025-26 End of Year Hackathon](https://www.kaggle.com/competitions/aiml-a-a-2025-26-end-of-year-hackathon/leaderboard) on Kaggle. We secured **2nd place** on the private leaderboard achieving a **MAP@10 score of 0.71**.

This multi-stage Information Retrieval (IR) pipeline combines classical lexical search (BM25), dense retrieval (Bi-Encoders), neural reranking (Cross-Encoders), and a final Learning-to-Rank model using XGBoost.

## Project Overview

The system is designed to retrieve the most relevant passages for a given query from a large corpus of documents. To achieve high performance while maintaining computational feasibility, the pipeline is divided into two main stages:

1.  **Stage 1: Retrieval & Fusion (Recall-oriented)**
    - Computes lexical scores using **BM25**.
    - Computes semantic scores using a **BGE Bi-Encoder** (`BAAI/bge-large-en-v1.5`).
    - Merges the top candidates from both methods using **Reciprocal Rank Fusion (RRF)** to generate a pool of Top-K candidate documents (default: 200) per query.
2.  **Stage 2: Neural Reranking & LTR (Precision-oriented)**
    - Evaluates the restricted Top-K pool using two powerful **Cross-Encoders**:
      - `BAAI/bge-reranker-v2-m3`
      - `mixedbread-ai/mxbai-rerank-base-v1`
    - Computes advanced statistical features (Z-scores, model agreement/conflict flags, term overlaps).
    - Uses an **XGBoost Ranker** (`XGBRanker`) to learn the optimal combination of these features and generate the final Top-10 ranking.

## Repository Structure

The core logic of the pipeline is distributed across the following scripts:

-   `build_dataset.py`: Extracts the features from the Top-K candidates pool and constructs the tabular dataset (e.g., `train_ltr_pairs.parquet`) required to train the XGBoost Ranker.
-   `train_xgboost.py`: Loads the extracted dataset, prepares the features, and trains the `XGBRanker` model, saving the best artifacts to `xgb_ltr_best.json`.
-   `xgb_ltr_best.json`: Contains the parameters and learned weights of our best-performing XGBoost model.
-   `baseline_map.py`: Evaluates the zero-shot baseline performance on the validation set using only manual heuristics (Cross-Encoder score weighting) without the XGBoost model.
-   `val_submission.py`: Evaluates the full pipeline (including the trained XGBoost LTR model) on the validation set, calculating the **MAP@10** metric and performing Grid Search for optimal feature weights.

## Execution Flow

To reproduce the workflow:

1.  **Prepare Data**: Ensure your Datasets (`collection.parquet`, `train.parquet`, etc.) are placed inside the `Data/` directory.
2.  **Build Training Set**: Run `build_dataset.py` to generate the `.parquet` training features.
3.  **Evaluate & Validate**: Run `val_submission.py` to test the full pipeline on your validation set (e.g., Grid Search for MAP@10).
4.  **Infer**: Run `run_submission.py` to generate the final `submission.csv`.

**💡 Note on Custom Training via `val_submission.py`:**
If you want to quickly train the XGBoost model on your own entirely custom dataset paths, you can technically use `val_submission.py` as a shortcut:
1. Change the `validation_url` inside `val_submission.py` to point to your new training `.parquet` file.
2. Uncomment the `build_dataset(...)` method call (around line 360).
3. Run the script. Let the Retrieval and Cross-Encoders compute the top candidates and save them. 
4. Once `build_dataset` finishes saving the `train_ltr_pairs.parquet` file to disk, you can stop/interrupt the execution of `val_submission.py`.
5. Finally, run `train_xgboost.py` to train the Learning-to-Rank model on your newly generated dataset!

## Dependencies

The project relies on standard data science libraries and modern IR frameworks:
-   `pandas`, `numpy`, `scikit-learn`
-   `xgboost`
-   `torch` (PyTorch)
-   `sentence-transformers`
-   `rank_bm25`
-   `nltk`
