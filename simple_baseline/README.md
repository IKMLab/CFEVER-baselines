# Simple Baseline for CFEVER
## Installation
```
cd simple_baseline
pip install -r requirements.txt
```

## Document Retrieval
Please refer to the steps in the [elasticsearch](https://github.com/elastic/elasticsearch) repository to run BM25 for document retrieval. We provide our results in the [simple_baseline/data/bm25_test](simple_baseline/data/bm25_test) directory.

## Sentence Retrieval
You can run the following example command to retrieve sentences from the documents:
```
PYTHONPATH=src python scripts/run_sentence_retrieval.py \
--data_dir data/bm25 \
--stage_1_doc_n 10
```

## Claim Verification
To perform claim verification, please run the following example command:
```
PYTHONPATH=src python scripts/run_claim_verification.py \
--data_dir data/bm25_test \
--sent_exp_name sent_chinese-bert-wwm-ext
```