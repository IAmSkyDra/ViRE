# VIRE: Vietnamese Information Retrieval Evaluation Toolkit

A lightweight, extensible toolkit for benchmarking lexical, dense, and hybrid retrieval on Vietnamese datasets.

> This repository accompanies the paper:     
> **Which Works Best for Vietnamese? A Practical Study of Information Retrieval ?Methods across Domains**  
> Under review for *ACL Rolling Review - October 2025*. 

---

## Why VIRE?

- Unified CLI to benchmark BM25, dense, and hybrid retrieval with fair, reproducible settings
- Dataset-agnostic schema — plug in any CSV/qrels following the standard specification
- Full pipeline: normalization &rarr; indexing → evaluation → error analysis → reporting

---

## Core Features

VIRE provides a comprehensive suite of features designed to support rigorous evaluation of information retrieval methods on Vietnamese datasets. Our toolkit encompasses the full experimental pipeline from data preprocessing to final reporting.

**Diverse Retrieval Methods**
 - Lexical: TF-IDF, BM25
 - Dense: OpenAI, Gemini, *Sentence-Transformers* (SBERT)
 - Hybrid: dense + lexical with Alpha fusion or *Reciprocal Rank Fusion* (RRF)

**Vietnamese-aware Preprocessing**
- Performs Unicode NFKC normalization, removes invisible characters and emojis, lowercases, and optionally segments Vietnamese words.
- Includes random and unique-context sampling with corpus deduplication.

**Evaluation and Reporting**
- Supports Precision@*k*, Recall@*k*, HitRate@*k*, MRR@*k*, MAP@*k*, nDCG@*k*, *R*-Precision, and First Relevant Rank (mean, median, found rate).
- Outputs include `metrics.json`, `ranks.json`, `embeddings`, and `index.faiss`.
- Markdown aggregation available via `scripts/summarize_report.py`.

**Error Analysis and Intersections.** Exports `errors/fail@K.csv`, computes cross-method failure intersections (`--save-intersection`), and provides per-query diagnostics (query, gold docs, retrieved docs, scores).

**Multiple Embedding Backends.** Supports OpenAI and Gemini APIs, or local SBERT models with automatic device selection and OOM-safe batching.

**Reusability and Extensibility.** Includes embedding and FAISS caching, flexible CLI, multi-gold qrels support, and clean interfaces for adding new retrievers or fusion modules.

**Easy Integration.** Standardized outputs for *Retrieval-Augmented Generation* (RAG)  and *Question-Answering* (QA) pipelines; supports ablation studies by swapping backends, tokenization, or fusion methods.

---

## Command-Line Interface (CLI)

VIRE is designed as a command-line tool to facilitate integration into research workflows and batch processing scripts. The complete interface provides fine-grained control over all experimental parameters.

```
Evaluate lexical/dense/hybrid retrieval for Vietnamese QA.

options:
  -h, --help            show this help message and exit
  --csv CSV             Input dataset file (CSV/JSONL) with columns or
                        keys: question, context (default: None)
  --qrels QRELS         Optional qrels file (CSV/TSV/JSONL) with columns:
                        qid/doc_id/rel (default: None)
  --qid-col QID_COL     Qrels column for query id (default: qid)
  --docid-col DOCID_COL
                        Qrels column for doc id (default: doc_id)
  --rel-col REL_COL     Qrels column for relevance (>0) (default: rel)
  --csv-qid-col CSV_QID_COL
                        Column name in main CSV/JSONL for qid (optional)
                        (default: None)
  --csv-docid-col CSV_DOCID_COL
                        Column name in main CSV/JSONL for doc_id
                        (optional) (default: None)
  --output-dir OUTPUT_DIR
                        Root folder to save results (default: outputs)
  --method {tfidf,bm25,dense,dense+tfidf,dense+bm25}
                        Retrieval method (default: None)
  --fusion {none,alpha,rrf}
                        Fusion for hybrid methods (dense+tfidf /
                        dense+bm25] (default: none)
  --alpha ALPHA         Alpha for score fusion (0..1) (default: 0.5)
  --rrf-k RRF_K         RRF constant k (>=1) (default: 60)
  --max-samples MAX_SAMPLES
                        Random subset size, e.g. 1000 (default: None)
  --sample-frac SAMPLE_FRAC
                        Random subset fraction, e.g. 0.1 (10%) (default:
                        None)
  --sample-seed SAMPLE_SEED
                        Random seed for sampling (default: 42)
  --prefer-unique       Prefer samples with unique contexts when sampling
                        (normalized) (default: False)
  --unique-col UNIQUE_COL
                        Column name used as uniqueness key for --prefer-
                        unique (default: context)
  --dedup-lower         Lowercase in normalization key for unique/dedup
                        (default: False)
  --dedup-remove-emoji  Strip emoji-like symbols in normalization key for
                        unique/dedup (default: False)
  --normalize-all       Normalize ALL questions/contexts with NFKC,
                        remove invisibles/controls, strip emoji, collapse
                        spaces, lowercase BEFORE snapshot/dedup/eval
                        (default: False)
  --bm25-k1 BM25_K1     BM25 Okapi k1 (default: 1.5)
  --bm25-b BM25_B       BM25 Okapi b (default: 0.75)
  --dense-backend {openai,gemini,sbert}
                        Dense embedding backend (default: openai)
  --embed-model EMBED_MODEL
                        OpenAI embedding model (default: text-
                        embedding-3-large)
  --gemini-model GEMINI_MODEL
                        Gemini embedding model (default: text-
                        embedding-004)
  --sbert-model SBERT_MODEL
                        Sentence-Transformers model (default: sentence-
                        transformers/all-MiniLM-L6-v2)
  --batch-size BATCH_SIZE
                        Embedding batch size (default: 128)
  --index-metric {ip,l2}
                        FAISS metric (recommend 'ip' with normalized
                        vectors) (default: ip)
  --force               Force rebuild embeddings and index (default:
                        False)
  --lower               Lowercase text before processing (kept for
                        backward-compat; ignored if --normalize-all)
                        (default: False)
  --ks KS               Comma-separated k values (default:
                        1,3,5,10,20,50,100)
  --show-size           Print dataset head + sizes (default: False)
  --progress            Show progress bars (default: False)
  --log-level {debug,info,warning,error}
                        Logger level (if runner uses logging_utils)
                        (default: info)
  --log-file LOG_FILE   Optional log file path (default: None)
  --list-backends       List available dense embedding backends and exit
                        (default: False)
  --dedup               Deduplicate identical contexts in the corpus
                        before indexing (queries remain unchanged)
                        (default: False)
  --error-k ERROR_K     Reference K used to mark a query as FAIL
                        (default: max(ks)). (default: None)
  --save-intersection   After run, save the intersection of fail@K across
                        ALL methods for this dataset. (default: False)
  --max-errors MAX_ERRORS
                        When printing previews, max rows to show.
                        (default: 30)
```

---

## Installation

Getting started with VIRE requires minimal setup. The toolkit is designed to work with standard Python environments and popular machine learning libraries.

```bash
git clone https://anonymous.4open.science/r/ViRE.git
cd ViRE
pip install -e .
pip install -U faiss-cpu sentence-transformers underthesea
```

Set environment variables for API-based backends:

```bash
export OPENAI_API_KEY=...     # for OpenAI embeddings
export GEMINI_API_KEY=...     # for Google Gemini embeddings
```

---

## Quickstart

This section demonstrates common usage patterns, from single-method evaluation to comprehensive benchmarking across multiple datasets and retrieval approaches.

### Single Runs

For quick experimentation or testing specific configurations:

```bash
# BM25 baseline
vi-retrieval-eval --csv data/demo.csv --method bm25 --output-dir outputs

# Dense retrieval using SBERT
vi-retrieval-eval --csv data/demo.csv --method dense \
	--dense-backend sbert --sbert-model AITeamVN/Vietnamese_Embedding_v2 \
	--output-dir outputs

# Hybrid (Alpha Fusion)
vi-retrieval-eval --csv data/demo.csv --method dense+bm25 \
	--fusion alpha --alpha 0.7 --output-dir outputs

# Hybrid (RRF Fusion)
vi-retrieval-eval --csv data/demo.csv --method dense+tfidf \
	--fusion rrf --rrf-k 60 --output-dir outputs
```

### Batch Runs for Multiple Datasets

For comprehensive evaluation across multiple datasets and methods, we provide a template script that automates the entire benchmarking process:

```bash
#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------------------
# Batch benchmarking script for VIRE
# Runs dense and hybrid methods (alpha, RRF) across multiple datasets
# Produces metrics, embeddings, and error reports (fail@K)
# -------------------------------------------------------------------------

DATASETS=("CSConDa" "EduCoQA" "VlogQA_2" "ViRe4MRC_v2")

BACKEND="sbert"
SBERT_MODEL="AITeamVN/Vietnamese_Embedding_v2"
BATCH=5
MAX_SAMPLES=1000
K_REF=20
OUTDIR="outputs"

for NAME in "${DATASETS[@]}"; do
	echo "Running evaluation for dataset: $NAME"

	# Select dataset + qrels mapping if applicable
	if [[ "$NAME" == "ZaloLegalQA" ]]; then
		CSV="data/ZaloLegalQA/dataset.csv"
		QRELS=(--qrels "data/ZaloLegalQA/all.jsonl"
					 --csv-qid-col qid
					 --csv-docid-col doc_id
					 --qid-col query-id
					 --docid-col corpus-id
					 --rel-col score)
	else
		CSV="data/${NAME}.csv"
		QRELS=()
	fi

	# 1. Dense baseline
	vi-retrieval-eval --csv "$CSV" "${QRELS[@]}" \
		--method dense --dense-backend "$BACKEND" --sbert-model "$SBERT_MODEL" \
		--batch-size "$BATCH" --max-samples "$MAX_SAMPLES" \
		--prefer-unique --dedup --progress --error-k "$K_REF" \
		--output-dir "$OUTDIR"

	# 2. Hybrid: Dense + TF-IDF (alpha fusion)
	vi-retrieval-eval --csv "$CSV" "${QRELS[@]}" \
		--method dense+tfidf --fusion alpha --alpha 0.7 \
		--dense-backend "$BACKEND" --sbert-model "$SBERT_MODEL" \
		--batch-size "$BATCH" --max-samples "$MAX_SAMPLES" \
		--prefer-unique --dedup --progress --error-k "$K_REF" \
		--output-dir "$OUTDIR"

	# 3. Hybrid: Dense + TF-IDF (RRF)
	vi-retrieval-eval --csv "$CSV" "${QRELS[@]}" \
		--method dense+tfidf --fusion rrf --rrf-k 60 \
		--dense-backend "$BACKEND" --sbert-model "$SBERT_MODEL" \
		--batch-size "$BATCH" --max-samples "$MAX_SAMPLES" \
		--prefer-unique --dedup --progress --error-k "$K_REF" \
		--output-dir "$OUTDIR"

	# 4. Hybrid: Dense + BM25 (alpha)
	vi-retrieval-eval --csv "$CSV" "${QRELS[@]}" \
		--method dense+bm25 --fusion alpha --alpha 0.7 \
		--dense-backend "$BACKEND" --sbert-model "$SBERT_MODEL" \
		--bm25-k1 1.5 --bm25-b 0.75 \
		--batch-size "$BATCH" --max-samples "$MAX_SAMPLES" \
		--prefer-unique --dedup --progress --error-k "$K_REF" \
		--output-dir "$OUTDIR"

	# 5. Hybrid: Dense + BM25 (RRF, save intersections)
	vi-retrieval-eval --csv "$CSV" "${QRELS[@]}" \
		--method dense+bm25 --fusion rrf --rrf-k 60 \
		--dense-backend "$BACKEND" --sbert-model "$SBERT_MODEL" \
		--bm25-k1 1.5 --bm25-b 0.75 \
		--batch-size "$BATCH" --max-samples "$MAX_SAMPLES" \
		--prefer-unique --dedup --progress --error-k "$K_REF" \
		--output-dir "$OUTDIR" --save-intersection

	echo "Done: $NAME"
	echo "---------------------------------------------"
done
```

---

## Datasets

Our benchmark encompasses diverse domains and query types to ensure comprehensive evaluation of retrieval methods. We curated a benchmark where each dataset reflects naturally occurring queries paired with domain-specific documents, ensuring both linguistic diversity and retrieval difficulty.

**Education**

- Includes authentic student questions on admissions and academic rules.  
- Datasets: EduCoQA (proposed), ViRHE4QA [1].

**Customer Support**

- Collected real conversations between customers and support agents.  
- Dataset: CSConDa (proposed).

**Legal**

- Covers statutory and regulatory retrieval tasks.  
- Datasets: [ALQAC](https://alqac.github.io), [Zalo Legal Text Retrieval](https://challenge.zalo.ai/portal/legal-text-retrieval).

**Healthcare**

- Medical QA covering diseases, drugs, and treatments.  
- Datasets: ViNewsQA [2], ViMedAQA [3].

**Lifestyle and Reviews**

- Informal, everyday Vietnamese queries.  
- Datasets: VlogQA [4], ViRe4MRC [5].

**Cross-domain Open Knowledge**

- General Vietnamese QA from Wikipedia.  
- Dataset: UIT-ViQuAD [6].

> Each dataset was standardized to 1,000 query–document pairs with duplicates removed and gold relevance remapped to the deduplicated corpus. The 1,000-sample subsets reported in the paper are located in `data/`, while raw datasets are available in `data/raw/*`.
---

## Data Format

VIRE adopts a simple, standardized data format that facilitates easy integration of new datasets while maintaining compatibility with existing IR evaluation frameworks.

```
qid,doc_id,question,context
q1,d12,"How to apply for admission?","Application procedures and requirements..."
q2,d15,"What are the graduation requirements?","Students must complete all required courses..."
```

Optional multi-gold qrels:

```
qid,doc_id,rel
q1,d12,1
q1,d87,1
```

For large corpora, FAISS indices and embeddings are automatically cached and reused.

---

## Outputs

VIRE generates comprehensive outputs for each experimental run, facilitating both immediate analysis and long-term result storage. All outputs follow a consistent directory structure for easy navigation and comparison.

```
outputs/<dataset>/<method>/
	├── metrics.json
	├── ranks.json
	├── errors/
	│   └── fail@K.csv
	├── index.faiss
	├── doc_embeddings.npy
	└── query_embeddings.npy
```

---

## Reports

To facilitate analysis and presentation of results across multiple experiments, VIRE includes a report generation system that produces publication-ready tables and visualizations.

```bash
python scripts/summarize_report.py \
	--outputs-root outputs \
	--datasets CSConDa \
	--metrics Precision@1,Recall@10,Recall@20,Recall@50,MRR@10 \
	--save VIRE_Report.md
```

**Available options:**
- `--outputs-root`: Root directory containing experimental results
- `--datasets`: Comma-separated list of dataset names
- `--metrics`: Comma-separated metric names (supports aliases like `p@10`, `r@20`)
- `--ndigits`: Number of decimal places for formatting (default: 4)
- `--percent`: Show metrics as percentages
- `--include`: Regex filter for method folder names

The script generates grouped tables with **bold** highlighting for best results and <u>underlined</u> values for second-best performance.

**Example Output:**

The generated report provides a comprehensive comparison across methods and datasets:

Method | precision@1 | recall@10 | recall@20 | recall@50 | mrr@10
--- | --- | --- | --- | --- | ---
tfidf | 15.70% | 38.50% | 47.20% | 59.50% | 22.49%
bm25 | 17.40% | 36.80% | 45.90% | 56.00% | 22.99%
**Dense model: openai-text-embedding-3-large** |  |  |  |  | 
  dense | 33.70% | 56.80% | 63.80% | 73.10% | 41.06%
  dense + tfidf (alpha) | <u>34.90%</u> | **60.40%** | **66.50%** | **75.90%** | <u>42.45%</u>
  dense + bm25 (alpha) | **36.40%** | <u>60.20%</u> | <u>66.20%</u> | <u>74.70%</u> | **43.60%**
  dense + tfidf (rrf) | 28.80% | 55.00% | 63.80% | 74.30% | 36.45%
  dense + bm25 (rrf) | 29.60% | 54.40% | 64.00% | 73.00% | 37.05%
**Dense model: sbert-bge-m3** |  |  |  |  | 
  dense | 30.80% | 53.90% | 61.00% | 69.90% | 37.98%
  dense + tfidf (alpha) | 33.10% | 57.00% | 63.80% | 73.10% | 40.67%
  dense + bm25 (alpha) | 33.90% | 56.90% | 63.90% | 72.80% | 40.97%
  dense + tfidf (rrf) | 28.40% | 54.20% | 63.90% | 71.60% | 35.82%
  dense + bm25 (rrf) | 28.60% | 53.70% | 62.80% | 71.10% | 36.24%
**Dense model: sbert-paraphrase-multilingual-MiniLM-L12-v2** |  |  |  |  | 
  dense | 11.80% | 30.00% | 39.10% | 49.50% | 16.71%
  dense + tfidf (alpha) | 19.60% | 45.30% | 51.40% | 62.50% | 27.05%
  dense + bm25 (alpha) | 18.90% | 45.40% | 51.60% | 61.70% | 26.18%
  dense + tfidf (rrf) | 17.70% | 43.80% | 54.30% | 64.90% | 25.03%
  dense + bm25 (rrf) | 17.50% | 43.80% | 53.30% | 64.80% | 24.63%
**Dense model: sbert-vietnamese-bi-encoder** |  |  |  |  | 
  dense | 15.70% | 34.90% | 41.70% | 53.40% | 21.09%
  dense + tfidf (alpha) | 22.20% | 46.00% | 52.40% | 63.00% | 28.95%
  dense + bm25 (alpha) | 22.70% | 45.70% | 52.80% | 62.50% | 28.93%
  dense + tfidf (rrf) | 19.10% | 44.20% | 55.00% | 65.80% | 26.34%
  dense + bm25 (rrf) | 20.00% | 44.90% | 54.30% | 65.30% | 26.98%
**Dense model: sbert-vietnamese-document-embedding** |  |  |  |  | 
  dense | 28.40% | 53.00% | 59.90% | 68.30% | 36.12%
  dense + tfidf (alpha) | 31.10% | 57.90% | 65.50% | 73.80% | 39.46%
  dense + bm25 (alpha) | 32.40% | 57.80% | 64.60% | 73.10% | 40.05%
  dense + tfidf (rrf) | 26.00% | 51.80% | 64.00% | 74.60% | 33.88%
  dense + bm25 (rrf) | 26.70% | 52.60% | 63.70% | 74.40% | 34.65%
**Dense model: sbert-Vietnamese_Embedding_V2** |  |  |  |  | 
  dense | 31.40% | 54.00% | 61.40% | 70.00% | 38.40%
  dense + tfidf (alpha) | 32.70% | 57.50% | 65.30% | 73.50% | 40.51%
  dense + bm25 (alpha) | 33.70% | 57.90% | 64.30% | 73.30% | 41.22%
  dense + tfidf (rrf) | 28.10% | 54.60% | 63.90% | 72.50% | 35.84%
  dense + bm25 (rrf) | 28.80% | 54.70% | 62.40% | 71.90% | 36.70%

---

## Code Structure

VIRE follows a modular architecture that separates concerns and enables easy extensibility:

```
src/vi_retrieval_eval/
├── cli.py              # Command-line interface and argument parsing
├── runner.py           # Main evaluation orchestrator
├── metrics.py          # IR evaluation metrics (P@k, R@k, MRR, nDCG, etc.)
├── lexical.py          # TF-IDF and BM25 implementations
├── dense_index.py      # FAISS indexing and dense retrieval
├── fusion.py           # Hybrid fusion methods (Alpha, RRF)
├── embeddings/         # Embedding backend implementations
│   ├── base.py         #   Registry and base interface
│   ├── openai_embed.py #   OpenAI API integration
│   ├── gemini_embed.py #   Google Gemini API integration
│   └── sbert_embed.py  #   Sentence-Transformers local models
├── qrels.py            # Relevance judgment handling
├── io_utils.py         # File I/O utilities (CSV, JSON, JSONL)
├── sampling.py         # Dataset sampling and unique selection
├── dedup.py            # Corpus deduplication
├── textnorm.py         # Vietnamese text normalization
├── tokenization.py     # Text preprocessing and tokenization
├── stats.py            # Dataset statistics computation
├── progress.py         # Progress bar utilities
└── logging_utils.py    # Logging configuration
```

**Key Design Patterns:**

- **Plugin Architecture**: Embedding backends register via decorators in `embeddings/base.py`
- **Factory Pattern**: `get_embedder()` creates instances based on string identifiers
- **Pipeline Processing**: CLI → Runner → Components → Metrics → Output
- **Caching Strategy**: Embeddings and FAISS indices cached by content hash
- **Error Isolation**: Each component handles failures gracefully with detailed error reporting

---

## Reproducibility

VIRE is designed with reproducibility as a core principle, ensuring that experimental results can be consistently replicated across different environments and time periods.

- Fixed random seeds for sampling and tie-breaking
- Stable mergesort for consistent ranking order
- Dependency management via `pyproject.toml`
- Cached embeddings and FAISS indices for efficient reruns
- Deterministic text normalization and preprocessing

---

## Extending VIRE

The toolkit is designed with extensibility in mind, allowing researchers to easily incorporate new retrieval methods, datasets, and evaluation approaches without modifying core functionality.

- **New retriever:** add under `src/vi_retrieval_eval/embeddings/` and register with `@register("your_name")` decorator
- **New dataset:** convert to CSV/qrels schema or create custom loader in `io_utils.py`
- **New fusion:** implement in `src/vi_retrieval_eval/fusion.py` and expose via `--fusion your_method`
- **New metrics:** add to `src/vi_retrieval_eval/metrics.py` following the existing pattern

---

## License

This project follows open science principles while respecting the licensing requirements of third-party datasets and dependencies.

- **Code:** MIT
- **New datasets:** CC BY-NC 4.0 (research & education only)
- **Third-party datasets:** original licenses apply

---

## Citation

If you use VIRE in your research, please cite our work:

```
@software{anonymous2025,
	title  = {Which Works Best for Vietnamese? A Practical Study of Information Retrieval Methods across Domains},
	author = {Anonymous},
	year   = {2025},
	url    = {https://anonymous.4open.science/r/ViRE/README.md}
}
```

## References

[1] T. P. P. Do, N. D. D. Cao, K. Q. Tran, and K. Van Nguyen, “R2GQA: retriever-reader-generator question answering system to support students understanding legal regulations in higher education”, *Artificial Intelligence and Law*, May 2025.

[2] K. Van Nguyen, T. Van Huynh, D.-V. Nguyen, A. G.-T. Nguyen, and N. L.-T. Nguyen, “New Vietnamese Corpus for Machine Reading Comprehension of Health News Articles”, *ACM Transactions on Asian and Low-Resource Language Information Processing*, vol. 21, no. 5, Sep. 2022.

[3] M.-N. Tran, P.-V. Nguyen, L. Nguyen, and D. Dinh, “ViMedAQA: A Vietnamese Medical Abstractive Question-Answering Dataset and Findings of Large Language Model”, in *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 4: Student Research Workshop)*, X. Fu and E. Fleisig, Eds. Bangkok, Thailand: Association for Computational Linguistics, Aug. 2024, pp. 252–260.

[4] T. Ngo, K. Dang, S. Luu, K. Nguyen, and N. Nguyen, “VlogQA: Task,
Dataset, and Baseline Models for Vietnamese Spoken-Based Machine
Reading Comprehension”, in *Proceedings of the 18th Conference of
the European Chapter of the Association for Computational Linguistics
(Volume 1: Long Papers)*, Y. Graham and M. Purver, Eds. St. Julian’s,
Malta: Association for Computational Linguistics, Mar. 2024, pp. 1310–
1324. 

[5] T. P. P. Do, N. D. D. Cao, N. T. Nguyen, T. V. Huynh, and K. V.
Nguyen, “Machine Reading Comprehension for Vietnamese Customer
Reviews: Task, Corpus and Baseline Models”, in *Proceedings of the 37th
Pacific Asia Conference on Language, Information and Computation*,
C.-R. Huang, Y. Harada, J.-B. Kim, S. Chen, Y.-Y. Hsu, E. Chersoni,
P. A, W. H. Zeng, B. Peng, Y. Li, and J. Li, Eds. Hong Kong,
China: Association for Computational Linguistics, Dec. 2023, pp. 24–35.

[6] K. Van Nguyen, D.-V. Nguyen, A. Gia-Tuan Nguyen, and N. Luu- Thuy Nguyen, “A Vietnamese Dataset for Evaluating Machine Reading Comprehension”, in *Proceedings of the 28th International Conference on Computational Linguistics*, D. Scott, N. Bel, and C. Zong, Eds. Barcelona, Spain (Online): International Committee on Computational Linguistics, Dec. 2020, pp. 2595–2605.
