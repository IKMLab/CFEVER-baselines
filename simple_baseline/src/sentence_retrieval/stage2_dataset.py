from typing import Tuple, Dict
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from transformers import BertTokenizer
from common.dataset.reader import JSONLineReader


@dataclass
class DataPack:
    """DataPack class for BERT dataset.

    It contains the train, dev, and test data.
    """

    train: pd.DataFrame
    dev: pd.DataFrame
    test: pd.DataFrame


class BERTDataset(Dataset):
    """BERTDataset class for BERT model.

    Every dataset should subclass it. All subclasses should override `__getitem__`,
    that provides the data and label on a per-sample basis.

    Args:
        data (pd.DataFrame): The data to be used for training, validation, or testing.
        tokenizer (BertTokenizer): The tokenizer to be used for tokenization.
        max_length (int, optional): The max sequence length for input to BERT model. Defaults to 128.
        topk (int, optional): The number of top evidence sentences to be used. Defaults to 5.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: BertTokenizer,
        max_length: int = 128,
        topk: int = 5,
    ):
        """__init__ method for BERTDataset"""
        self.data = data.fillna("")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.topk = topk
        self.reader = JSONLineReader()

    def __len__(self):
        return len(self.data)


class SentRetrievalBERTDataset(BERTDataset):
    """AicupTopkEvidenceBERTDataset class for AICUP dataset with top-k evidence sentences."""

    def __getitem__(
        self,
        idx: int,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        item = self.data.iloc[idx]
        sentA = item["claim"]
        sentB = item["text"]

        # concat_claim_evidence = " [SEP] ".join([sentA, sentB])

        concat = self.tokenizer(
            sentA,
            sentB,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        concat_ten = {k: torch.tensor(v) for k, v in concat.items()}
        if "label" in item:
            concat_ten["labels"] = torch.tensor(item["label"])
        return concat_ten
