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


## Our simple baseline
Plase refer to the [simple_baseline](simple_baseline) folder and check the [README.md](simple_baseline/README.md) for more details.

