# %%
# -*- coding: utf-8 -*-
"""
File handler module
===================

This module provides functions for reading and writing files.


.. centered:: This script provides functions for reading and writing files.

.. codeauthor:: Yi-Ting Li <yt.li.public@gmail.com>
"""
from common.util.logger import logger as L
import re
import json
import re
from functools import partial, wraps
from pathlib import Path
from typing import Callable, Dict, List, Union

import pandas as pd
import numpy as np
from opencc import OpenCC
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True, verbose=0)

# from common.util.logger import logger as L
import logging

L = logging.getLogger(__name__)


def cache_file(func: Callable[..., pd.DataFrame]) -> Callable:
    """cache_file decorator

    If the file_path exists, read from file_path.
    If the file_path not exists, run func and write to file_path.

    Args:
        func (Callable[..., pd.DataFrame]): The function to be decorated.

    Warning:
        The function to be decorated must return a pandas DataFrame.

    Returns:
        callable: The decorated function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get cache file_path
        file_path = kwargs.pop("cache", None)

        # If file_path is None, run func
        if file_path is None:
            return func(*args, **kwargs)

        # Check file_path exists
        if not file_path.exists():
            L.info(
                f"File {file_path} not exists, run {func.__name__} and write to {file_path}"
            )
            file_path.parent.mkdir(parents=True, exist_ok=True)
            # If not exists, run func and write to file_path
            result = func(*args, **kwargs)
            # Save result to file_path
            save_fn = getattr(
                result,
                f"to_{file_path.suffix[1:] if file_path.suffix != '.pkl' else 'pickle'}",
            )
            # If suffix is csv, set index=False
            if file_path.suffix == ".csv":
                save_fn = partial(save_fn, index=False)
            # Save result to file_path
            save_fn(file_path)
            return result

        # If exists, read from file_path
        L.info(f"Read cache from {file_path}")
        return getattr(
            pd,
            f"read_{file_path.suffix[1:] if file_path.suffix != '.pkl' else 'pickle'}",
        )(file_path)

    return wrapper


@cache_file
def read_jsonl_file(file_path: Union[Path, str]) -> pd.DataFrame:
    """read_jsonl_file read jsonl file and return a pandas DataFrame.

    Args:
        file_path (Union[Path, str]): The jsonl file path.
        cache(Union[Path, str], optional): The cache file path. Defaults to None.
            If cache is None, return the result directly.

    Returns:
        pd.DataFrame: The jsonl file content.

    Example:
        >>> read_jsonl_file("data/extracted/train.jsonl")
               id            label  ... predicted_label                                      evidence_list
        0    3984          refutes  ...         REFUTES  [城市規劃是城市建設及管理的依據 ， 位於城市管理之規劃 、 建設 、 運作三個階段之首 ，...
        ..    ...              ...  ...             ...                                                ...
        945  3042         supports  ...         REFUTES  [北歐人相傳每當雷雨交加時就是索爾乘坐馬車出來巡視 ， 因此稱呼索爾為 “ 雷神 ” 。, ...

        [946 rows x 10 columns]
    """
    with open(file_path, "r") as json_file:
        json_list = list(json_file)
    out = [json.loads(json_str) for json_str in json_list]
    return pd.DataFrame(out)


@cache_file
def read_jsonl_dir(dir_path: Union[Path, str]) -> pd.DataFrame:
    """read_jsonl_dir read jsonl dir and return a pandas DataFrame.

    This function will read all jsonl files in the dir_path and concat them.

    Args:
        dir_path (Union[Path, str]): The jsonl dir path.
        cache(Union[Path, str], optional): The cache file path. Defaults to None.
            If cache is None, return the result directly.

    Returns:
        pd.DataFrame: The jsonl dir content.

    Example:
        >>> read_jsonl_dir("data/extracted_dir/")
               id            label  ... predicted_label                                      evidence_list
        0    3984          refutes  ...         REFUTES  [城市規劃是城市建設及管理的依據 ， 位於城市管理之規劃 、 建設 、 運作三個階段之首 ，...
        ..    ...              ...  ...             ...                                                ...
        945  3042         supports  ...         REFUTES  [北歐人相傳每當雷雨交加時就是索爾乘坐馬車出來巡視 ， 因此稱呼索爾為 “ 雷神 ” 。, ...

        [946 rows x 10 columns]
    """
    data = []
    for file in Path(dir_path).glob("*.jsonl"):
        data.append(read_jsonl_file(file))
    return pd.concat(data)


@cache_file
def join_with_full_wiki_pages(
    df: pd.DataFrame,
    wiki_pages: pd.DataFrame,
    dont_convert: bool = False,
) -> pd.DataFrame:
    """join_with_wiki_pages join the dataset with full wiki pages.

    This function will **first** extract the evidence from the dataframe `df`,
    then join the dataset with the wiki pages by the wiki_page id.

    Note:
        After join, the dataframe `df` will have 2 new columns: `lines`, `text`,
        and the `evidence` column will be replaced by the evidence text, i.e. the
        3rd item in the evidence column.

    Args:
        df (pd.DataFrame): The dataset with evidence.
        wiki_pages (pd.DataFrame): The wiki pages dataframe
        dont_convert (bool, optional): If True, don't convert to traditional chinese. Defaults to False.
        cache(Union[Path, str], optional): The cache file path. Defaults to None.

    Returns:
        pd.DataFrame: The dataset with wiki pages.
    """
    wiki_pages = wiki_pages.copy()

    def extract_evidence(df: pd.DataFrame) -> List[str]:
        wiki_ids = []
        for _, data in df.iterrows():
            try:
                evidence = data["evidence"][0][0]
                if len(evidence) != 4 or evidence[2] is None:
                    raise TypeError
            except (IndexError, TypeError):
                wiki_id = None
            else:
                wiki_id = evidence[2]
            wiki_ids.append(str(wiki_id))
        return wiki_ids

    df["evidence"] = extract_evidence(df)
    df = df.astype(str)
    wiki_pages = wiki_pages.astype(str)

    merged = (
        pd.merge(
            left=df,
            right=wiki_pages,
            how="left",
            left_on="evidence",
            right_on="id",
        )
        .drop(columns=["id_y"])
        .rename(columns={"id_x": "id"})
    )

    merged = merged.fillna("")

    if not dont_convert:
        cc = OpenCC("s2tw")
        target_columns = ["text", "lines", "claim", "evidence"]
        merged[target_columns] = merged[target_columns].applymap(
            lambda x: cc.convert(x)
        )

    return merged


@cache_file
def generate_evidence_to_wiki_pages_mapping(
    wiki_pages: pd.DataFrame,
) -> Dict[str, Dict[int, str]]:
    """generate_wiki_pages_dict generate a mapping from evidence to wiki pages by evidence id.

    Args:
        wiki_pages (pd.DataFrame): The wiki pages dataframe
        cache(Union[Path, str], optional): The cache file path. Defaults to None.
            If cache is None, return the result directly.

    Returns:
        pd.DataFrame:
    """
    # copy wiki_pages
    wiki_pages = wiki_pages.copy()

    # generate parse mapping
    L.info("Generate parse mapping")
    wiki_pages["evidence_map"] = wiki_pages["lines"].parallel_map(
        lambda x: dict(re.findall(r"(\d+)\t([^\t]+)", x)),
    )
    # generate id to evidence_map mapping
    L.info("Transform to id to evidence_map mapping")
    mapping = dict(
        zip(
            wiki_pages["id"].to_list(),
            wiki_pages["evidence_map"].to_list(),
        )
    )
    # release memory
    del wiki_pages
    return mapping


@cache_file
def join_with_topk_evidence(
    df: pd.DataFrame,
    wiki_pages: pd.DataFrame,
    mode: str = None,
    topk: int = 5,
) -> pd.DataFrame:
    """join_with_topk_evidence join the dataset with topk evidence.

    Note:
        After extraction, the dataset will be like this:
               id     label         claim                           evidence            evidence_list
        0    4604  supports       高行健...     [[[3393, 3552, 高行健, 0], [...  [高行健 （ ）江西赣州出...
        ..    ...       ...            ...                                ...                     ...
        945  2095  supports       美國總...  [[[1879, 2032, 吉米·卡特, 16], [...  [卸任后 ， 卡特積極參與...
        停各种战争及人質危機的斡旋工作 ， 反对美国小布什政府攻打伊拉克...

        [946 rows x 5 columns]

    Args:
        df (pd.DataFrame): The dataset with evidence.
        wiki_pages (pd.DataFrame): The wiki pages dataframe
        topk (int, optional): The topk evidence. Defaults to 5.
        cache(Union[Path, str], optional): The cache file path. Defaults to None.
            If cache is None, return the result directly.

    Returns:
        pd.DataFrame: The dataset with topk evidence_list.
            The `evidence_list` column will be: List[str]
    """
    mapping = generate_evidence_to_wiki_pages_mapping_new(
        wiki_pages,
    )

    # format evidence column to List[List[Tuple[str, str, str, str]]]
    L.info("Format evidence column")
    df["evidence"] = df["evidence"].parallel_map(
        lambda x: (
            [[x]]
            if not isinstance(x[0], list)
            else [x] if not isinstance(x[0][0], list) else x
        )
    )
    L.info("Extract evidence_list")
    if mode == "eval":
        # extract evidence
        df["evidence_list"] = df["predicted_evidence"].parallel_map(
            lambda x: (
                [
                    mapping.get(evi_id, {}).get(str(evi_idx), "")
                    for evi_id, evi_idx in x  # for each evidence list
                ][:topk]
                if isinstance(x, list)
                else []
            )
        )
        print(df["evidence_list"][:5])
    else:
        # extract evidence
        df["evidence_list"] = df["evidence"].parallel_map(
            lambda x: (
                [
                    (
                        " ".join(
                            [  # join evidence
                                mapping.get(evi_id, {}).get(str(evi_idx), "")
                                for _, _, evi_id, evi_idx in evi_list
                            ]
                        )
                        if isinstance(evi_list, list)
                        else ""
                    )
                    for evi_list in x  # for each evidence list
                ][:1]
                if isinstance(x, list)
                else []
            )
        )

    return df


@cache_file
def generate_evidence_to_wiki_pages_mapping_new(
    wiki_pages: pd.DataFrame,
) -> Dict[str, Dict[int, str]]:
    """generate_wiki_pages_dict generate a mapping from evidence to wiki pages by evidence id.

    Args:
        wiki_pages (pd.DataFrame): The wiki pages dataframe
        cache(Union[Path, str], optional): The cache file path. Defaults to None.
            If cache is None, return the result directly.

    Returns:
        pd.DataFrame:
    """

    def make_dict(x):
        result = {}
        sentences = re.split(r"\n(?=[0-9])", x)
        for sent in sentences:
            splitted = sent.split("\t")
            if len(splitted) < 2:
                return result
            result[splitted[0]] = splitted[1]
        return result

    # copy wiki_pages
    wiki_pages = wiki_pages.copy()

    # generate parse mapping
    L.info("Generate parse mapping")
    wiki_pages["evidence_map"] = wiki_pages["lines"].parallel_map(make_dict)
    # generate id to evidence_map mapping
    L.info("Transform to id to evidence_map mapping")
    mapping = dict(
        zip(
            wiki_pages["id"].to_list(),
            wiki_pages["evidence_map"].to_list(),
        )
    )
    # release memory
    del wiki_pages
    return mapping


@cache_file
def pair_with_wiki_sentences(
    wiki_pages: pd.DataFrame,
    df: pd.DataFrame,
    mapping: Dict,
) -> pd.DataFrame:
    """Only for creating train sentences."""
    # mapping = generate_evidence_to_wiki_pages_mapping_new(wiki_pages, )
    claims = []
    sentences = []
    labels = []
    converter = OpenCC("t2s.json")
    st_converter = OpenCC("s2t.json")

    def do_st_corrections(text):
        simplified = converter.convert(text)
        return st_converter.convert(simplified)

    # positive
    for i in range(len(df)):
        if df["label"].iloc[i] == "NOT ENOUGH INFO":
            continue
        claim = df["claim"].iloc[i]
        evidence_sets = df["evidence"].iloc[i]
        for evidence_set in evidence_sets:
            sents = []
            for evidence in evidence_set:
                page = evidence[2].replace(" ", "_")
                if page == "臺灣海峽危機#第二次臺灣海峽危機（1958）":
                    continue
                sent_idx = str(evidence[3])
                try:
                    sents.append(mapping[page][sent_idx])
                except KeyError:
                    sents.append(mapping[do_st_corrections(page)][sent_idx])
            # sents = [
            #     mapping[evidence[2].replace(" ", "_")][]
            #     for evidence in evidence_set
            # ]
            whole_evidence = " ".join(sents)

            claims.append(claim)
            sentences.append(whole_evidence)
            labels.append(1)

    # Calculate negative ratio
    num_all_candidates = 0
    for i in range(len(df)):
        if df["label"].iloc[i] == "NOT ENOUGH INFO":
            continue
        predicted_pages = df["predicted_pages"][i]
        for page in predicted_pages:
            page = page.replace(" ", "_")
            if page == "臺灣海峽危機#第二次臺灣海峽危機（1958）":
                continue
            try:
                num_all_candidates += len(mapping[page])
            except KeyError:
                true_pages = get_title_from_evidence(df["evidence"][i])
                if page in true_pages:
                    raise KeyError(f"{page} is not in our Wiki db but in true pages.")

    a_magic_number = 1.6
    negative_ratio = a_magic_number * len(labels) / num_all_candidates
    # `a_magic_number` needs to be tuned.
    print(f"Using `negative ratio`: {negative_ratio}")

    # negative
    for i in range(len(df)):
        if df["label"].iloc[i] == "NOT ENOUGH INFO":
            continue
        claim = df["claim"].iloc[i]

        evidence_set = set(
            [
                (evidence[2], evidence[3])
                for evidences in df["evidence"][i]
                for evidence in evidences
            ]
        )
        predicted_pages = df["predicted_pages"][i]
        for page in predicted_pages:
            page = page.replace(" ", "_")
            # ('城市規劃', sent_idx)
            try:
                page_sent_id_pairs = [(page, k) for k in mapping[page]]
            except KeyError:
                # print(f"{page} is not in our Wiki db.")
                continue

            for pair in page_sent_id_pairs:
                if pair in evidence_set:
                    continue
                text = mapping[page][pair[1]]
                # `np.random.rand(1) <= 0.05`: Control not to add too many negative samples
                if text != "" and np.random.rand(1) <= negative_ratio:
                    claims.append(claim)
                    sentences.append(text)
                    labels.append(0)

    return pd.DataFrame({"claim": claims, "text": sentences, "label": labels})


def pair_with_wiki_sentences_eval(
    wiki_pages: pd.DataFrame,
    df: pd.DataFrame,
    mapping: Dict,
    is_testset: bool = False,
) -> pd.DataFrame:
    """Only for creating dev and test sentences."""
    # mapping = generate_evidence_to_wiki_pages_mapping_new(wiki_pages, )
    claims = []
    sentences = []
    evidence = []
    predicted_evidence = []

    # negative
    for i in range(len(df)):
        # if df["label"].iloc[i] == "NOT ENOUGH INFO":
        #     continue
        claim = df["claim"].iloc[i]

        predicted_pages = df["predicted_pages"][i]
        for page in predicted_pages:
            page = page.replace(" ", "_")
            # ('城市規劃', sent_idx)
            try:
                page_sent_id_pairs = [(page, k) for k in mapping[page]]
            except KeyError:
                # print(f"{page} is not in our Wiki db.")
                continue

            for pair in page_sent_id_pairs:
                text = mapping[page][pair[1]]
                if text != "":
                    claims.append(claim)
                    sentences.append(text)
                    if not is_testset:
                        evidence.append(df["evidence"].iloc[i])
                    predicted_evidence.append([pair[0], int(pair[1])])

    return pd.DataFrame(
        {
            "claim": claims,
            "text": sentences,
            "evidence": evidence if not is_testset else None,
            "predicted_evidence": predicted_evidence,
        }
    )


def get_title_from_evidence(evidence):
    titles = []
    for evidence_set in evidence:
        if len(evidence_set) == 4 and evidence_set[2] is None:
            return []
        for evidence_sent in evidence_set:
            titles.append(evidence_sent[2])
    return list(set(titles))
