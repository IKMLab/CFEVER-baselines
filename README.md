# CFEVER-baselines
In the CFEVER paper, we test the CFEVER test set using the following baselines:
- BEVERS ([DeHaven and Scott, 2023](https://aclanthology.org/2023.fever-1.6/))
- Stammbach ([Stammbach, 2021](https://aclanthology.org/2021.fever-1.2/))
- Our simple baseline
- ChatGPT (GPT-3.5 and GPT-4)

For the first two baselines, please refer to their source code:
- [Source code link for BEVERS](https://github.com/mitchelldehaven/bevers)
- [Source code link for Stammbach](https://github.com/dominiksinsaarland/document-level-fever)

## Download the CFEVER dataset
Please go to the [CFEVER-data](https://github.com/IKMLab/CFEVER-data) repository to download the our dataset.

## Get Started
```
git clone https://github.com/IKMLab/CFEVER-baselines.git
cd CFEVER-baselines
```

## Quick Introduction to the CFEVER task
CFEVER is a Chinese Fact Extraction and VERification dataset. Similar to [FEVER (Thorne et al., 2018)](https://aclanthology.org/N18-1074/), the CFEVER task is to verify the veracity of a given claim while providing evidence from the Chinese Wikipedia (we provide our processed version in the [CFEVER-data](https://github.com/IKMLab/CFEVER-data) repository). Therefore, the task is split into three sub-tasks:
1. Document Retrieval: Retrieve relevant documents from the Chinese Wikipedia.
2. Sentence Retrieval: Select relevant sentences from the retrieved documents. 
3. Claim Verification: Determine whether the claim is “Supports”, “Refutes”, or “Not Enough Info.” Generally, in this stage, a model performs claim verification based on the provided claim in the dataset and the selected sentences from the stage 2 (sentence retrieval).

## Installation
```
pip install -r requirements.txt
```

## Our simple baseline
Plase refer to the [simple_baseline](simple_baseline) folder and check the [README.md](simple_baseline/README.md) for more details.

## Evaluations
### Document Retrieval
To evaluate document retrieval, you need to pass two paths to the script `eval_doc_example.py`:
- `$GOLD_FILE`: the path to the file with gold answers in the `jsonl` format.
- `$DOC_PRED_FILE`: the path to the file with predicted documents in the `jsonl` format.
```
python eval_doc_retrieval.py \
--source_file $GOLD_FILE \
--doc_pred_file $DOC_PRED_FILE
```
The example command is shown below:
```
python eval_doc_retrieval.py \
--source_file simple_baseline/data/dev.jsonl \
--doc_pred_file simple_baseline/data/bm25/dev_doc10.jsonl
```
Note that our evaluation of document retrieval aligns with the way of BEVERS. See [BEVERS's code](https://github.com/mitchelldehaven/bevers/blob/main/src/eval/measure_tfidf.py).
- You can also try to evaluate with the first k predicted pages by setting the `--top_k` parameter. For example, `--top_k 10` will evaluate the first 10 predicted pages.

### Sentence Retrieval and Claim Verification
We follow the same evaluation script of [fever-scorer](https://github.com/sheffieldnlp/fever-scorer) and add some parameters to run the script:
```
python eval_sent_retrieval_rte.py \
--gt_file $GOLD_FILE \
--submission_file $PRED_FILE
```
where `$GOLD_FILE` is the path to the file with gold answers in the `jsonl` format and `$PRED_FILE` is the path to the file with predicted answers in the `jsonl` format. The example command is shown below:
```
python eval_sent_retrieval_rte.py \
--gt_file simple_baseline/data/dev.jsonl \
--submission_file simple_baseline/data/dumb_dev_pred.jsonl
```
The script will output the scores of sentence retrieval:
- Precision
- Recall
- F1-score

and the scores of claim verification:
- Accuracy (printed as `Label accuracy`)
- FEVER Score (printed as `Strict accuracy`)

## Reference
If you find our work useful, please cite our paper.
```
@article{Lin_Lin_Yeh_Li_Hu_Hsu_Lee_Kao_2024,
    title = {CFEVER: A Chinese Fact Extraction and VERification Dataset},
    author = {Lin, Ying-Jia and Lin, Chun-Yi and Yeh, Chia-Jen and Li, Yi-Ting and Hu, Yun-Yu and Hsu, Chih-Hao and Lee, Mei-Feng and Kao, Hung-Yu},
    doi = {10.1609/aaai.v38i17.29825},
    journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
    month = {Mar.},
    number = {17},
    pages = {18626-18634},
    url = {https://ojs.aaai.org/index.php/AAAI/article/view/29825},
    volume = {38},
    year = {2024},
    bdsk-url-1 = {https://ojs.aaai.org/index.php/AAAI/article/view/29825},
    bdsk-url-2 = {https://doi.org/10.1609/aaai.v38i17.29825}
}
```
