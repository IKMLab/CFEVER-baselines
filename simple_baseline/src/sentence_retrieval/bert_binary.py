from pathlib import Path
from typing import Tuple
import json

import pandas as pd
import numpy as np

import torch
from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_scheduler,
)
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from common.util import file_handler as FH
from common.util.random import SimpleRandom
from common.util.args_handler import save_args_to_json
from common.dataset.reader import JSONLineReader
from common.util.logger import logger as L
from sentence_retrieval.stage2_dataset import SentRetrievalBERTDataset, DataPack


def read_wiki_pages(wiki_pages_dir: str) -> pd.DataFrame:
    """Load wiki pages of the AICUP 2023 competition
    Cache at `data/wiki_pages.pkl` can be loaded faster after the second time

    Args:
        wiki_pages_dir (str): directory of the wiki pages
        (wiki_pages_dir should contain several files in .jsonl format)

    Returns:
        pd.DataFrame: DataFrame of the wiki pages
    """
    cache_file = "data/wiki_pages.pkl"
    if Path(cache_file).exists():
        df = pd.read_pickle(cache_file)
    else:
        df = []
        jsonl_files = Path(wiki_pages_dir).glob("*.jsonl")
        for file in jsonl_files:
            with open(file, "r") as f:
                tmp = [json.loads(line) for line in f]
            df.append(pd.DataFrame(tmp))
        df = pd.concat(df)

        # save wiki-pages to cache
        df.to_pickle(cache_file)

    return df


def prepare_data(data_paths: list) -> DataPack:
    """Load train.jsonl, dev.jsonl, and test.jsonl into a dataclass object.

    Args:
        data_paths (list): paths of the train.jsonl, dev.jsonl, and test.jsonl

    Returns:
        DataPack: a pack containing train, dev, and test data in pd.DataFrame
    """
    reader = JSONLineReader()
    return DataPack(
        train=pd.DataFrame(reader.read(data_paths[0])),
        dev=pd.DataFrame(reader.read(data_paths[1])),
        test=pd.DataFrame(reader.read(data_paths[2])),
    )


def evidence_macro_precision(
    instance: dict, top_rows: pd.DataFrame
) -> Tuple[float, float]:
    """Calculate precision for sentence retrieval
    This function is modified from fever-scorer.
    https://github.com/sheffieldnlp/fever-scorer/blob/master/src/fever/scorer.py

    Args:
        instance (dict): a row of the dev set (dev.jsonl) of test set (test.jsonl)
        top_rows (pd.DataFrame): our predictions with the top probabilities

        IMPORTANT!!!
        instance (dict) should have the key of `evidence`.
        top_rows (pd.DataFrame) should have a column `predicted_evidence`.

    Returns:
        Tuple[float, float]:
        [1]: relevant and retrieved (numerator of precision)
        [2]: retrieved (denominator of precision)
    """
    this_precision = 0.0
    this_precision_hits = 0.0

    if instance["label"].upper() != "NOT ENOUGH INFO":
        all_evi = [
            [e[2], e[3]] for eg in instance["evidence"] for e in eg if e[3] is not None
        ]
        claim = instance["claim"]
        predicted_evidence = top_rows[top_rows["claim"] == claim][
            "predicted_evidence"
        ].tolist()

        for prediction in predicted_evidence:
            if prediction in all_evi:
                this_precision += 1.0
            this_precision_hits += 1.0

        return (
            (this_precision / this_precision_hits) if this_precision_hits > 0 else 1.0
        ), 1.0

    return 0.0, 0.0


def evidence_macro_recall(
    instance: dict, top_rows: pd.DataFrame
) -> Tuple[float, float]:
    """Calculate recall for sentence retrieval
    This function is modified from fever-scorer.
    https://github.com/sheffieldnlp/fever-scorer/blob/master/src/fever/scorer.py

    Args:
        instance (dict): a row of the dev set (dev.jsonl) of test set (test.jsonl)
        top_rows (pd.DataFrame): our predictions with the top probabilities

        IMPORTANT!!!
        instance (dict) should have the key of `evidence`.
        top_rows (pd.DataFrame) should have a column `predicted_evidence`.

    Returns:
        Tuple[float, float]:
        [1]: relevant and retrieved (numerator of recall)
        [2]: relevant (denominator of recall)
    """
    # We only want to score F1/Precision/Recall of recalled evidence for NEI claims
    if instance["label"].upper() != "NOT ENOUGH INFO":
        # If there's no evidence to predict, return 1
        if len(instance["evidence"]) == 0 or all([len(eg) == 0 for eg in instance]):
            return 1.0, 1.0

        claim = instance["claim"]

        predicted_evidence = top_rows[top_rows["claim"] == claim][
            "predicted_evidence"
        ].tolist()

        for evidence_group in instance["evidence"]:
            evidence = [[e[2], e[3]] for e in evidence_group]
            if all([item in predicted_evidence for item in evidence]):
                # We only want to score complete groups of evidence. Incomplete groups are worthless.
                return 1.0, 1.0
        return 0.0, 1.0
    return 0.0, 0.0


def evaluate_retrieval(
    probs: np.ndarray,
    df_evidences: pd.DataFrame,
    ground_truths: pd.DataFrame,
    top_n: int = 5,
    cal_scores: bool = True,
    save_name: str = None,
) -> dict[float, float, float]:
    """Calculate the scores of sentence retrieval

    Args:
        probs (np.ndarray): probabilities of the candidate retrieved sentences
        df_evidences (pd.DataFrame): the candiate evidence sentences paired with claims
        ground_truths (pd.DataFrame): the loaded data of dev.jsonl or test.jsonl
        top_n (int, optional): the number of the retrieved sentences. Defaults to 2.

    Returns:
        dict[float, float, float]: F1 score, precision, and recall
    """
    df_evidences["prob"] = probs
    top_rows = (
        df_evidences.groupby("claim")
        .apply(lambda x: x.nlargest(top_n, "prob"))
        .reset_index(drop=True)
    )

    if cal_scores:
        macro_precision = 0
        macro_precision_hits = 0
        macro_recall = 0
        macro_recall_hits = 0

        for i, instance in enumerate(ground_truths):
            macro_prec = evidence_macro_precision(instance, top_rows)
            macro_precision += macro_prec[0]
            macro_precision_hits += macro_prec[1]

            macro_rec = evidence_macro_recall(instance, top_rows)
            macro_recall += macro_rec[0]
            macro_recall_hits += macro_rec[1]

        pr = (
            (macro_precision / macro_precision_hits)
            if macro_precision_hits > 0
            else 1.0
        )
        rec = (macro_recall / macro_recall_hits) if macro_recall_hits > 0 else 0.0
        f1 = 2.0 * pr * rec / (pr + rec)

    if save_name is not None:
        # write doc7_sent5 file
        with open(f"{save_name}", "w") as f:
            for instance in ground_truths:
                claim = instance["claim"]
                predicted_evidence = top_rows[top_rows["claim"] == claim][
                    "predicted_evidence"
                ].tolist()
                instance["predicted_evidence"] = predicted_evidence
                f.write(json.dumps(instance, ensure_ascii=False) + "\n")

    if cal_scores:
        return {"F1 score": f1, "Precision": pr, "Recall": rec}


def set_lr_scheduler(optimizer, num_training_steps: int):
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps * 0.1),
        num_training_steps=num_training_steps,
    )
    return lr_scheduler


def get_predicted_probs(model, dataloader, device) -> np.ndarray:
    """Inference script to get probabilites for the candidate evidence sentences

    Args:
        model: the one from HuggingFace Transformers
        dataloader: devset or testset in torch dataloader

    Returns:
        np.ndarray: probabilites of the candidate evidence sentences
    """
    model.eval()
    probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            probs.extend(torch.softmax(logits, dim=1)[:, 1].tolist())

    return np.array(probs)


def save_checkpoint(model, ckpt_dir: str, current_step: int):
    torch.save(model.state_dict(), f"{ckpt_dir}/model.{current_step}.pt")


def load_model(model, ckpt_dir: str, step: int):
    model.load_state_dict(torch.load(f"{ckpt_dir}/model.{step}.pt"))
    return model


def do_eval(
    model,
    device,
    dev_loader,
    df_evidences,
    ground_truths,
    top_n,
    validation_history,
    current_steps,
    writer,
    ckpt_dir,
):
    print("Start validation")
    probs = get_predicted_probs(model, dev_loader, device)

    val_results = evaluate_retrieval(
        probs=probs,
        df_evidences=df_evidences,
        ground_truths=ground_truths,
        top_n=top_n,
    )
    print(val_results)
    validation_history[current_steps] = val_results["F1 score"]
    # log each metric separately to TensorBoard
    for metric_name, metric_value in val_results.items():
        writer.add_scalar(
            f"dev_{metric_name}",
            metric_value,
            current_steps,
        )
    save_checkpoint(model, ckpt_dir, current_steps)


def sent_retrieval_main(
    data_dir: str,
    save_dir: str,
    sent_exp_name: str,
    model_name: str,
    num_epochs: int,
    learning_rate: float,
    train_batch_size: int,
    test_batch_size: int,
    negative_ratio: float,
    validation_step: int,
    seed: int,
    doc_n: int,
    top_n: int,
    do_test: bool = True,
    do_test_only: bool = False,
    test_set: list = None,
    pretrained_model_name=None,
    special_ckpt_name=None,
    special_ckpt_dir=None,
):
    data_name = data_dir.split("/")[-1]
    SimpleRandom.set_seed(seed)
    exp_dir = (
        f"sent_retrieval/{data_name}/{sent_exp_name}/"
        + f"e{num_epochs}_bs{train_batch_size}_"
        + f"{learning_rate}_neg{negative_ratio}_top{top_n}"
    )

    log_dir = "logs/" + exp_dir
    ckpt_dir = "checkpoints/" + exp_dir
    if not Path(ckpt_dir).exists():
        Path(ckpt_dir).mkdir(parents=True)

    train_path = f"{data_dir}/train_doc{doc_n}.jsonl"
    dev_path = f"{data_dir}/dev_doc{doc_n}.jsonl"
    test_path = f"{data_dir}/test_doc{doc_n}.jsonl"

    # save_dir = f"{data_dir}/{sent_exp_name}"
    # Path(save_dir).mkdir(parents=False, exist_ok=False)
    # save_args_to_json(args, f"{save_dir}/args.json")

    L.info("Preparing fever data...")
    print(f"Loading {[train_path, dev_path, test_path]}...")

    datapack = prepare_data([train_path, dev_path, test_path])
    wiki_pages = read_wiki_pages("data/wiki-pages")
    wiki_mapping = FH.generate_evidence_to_wiki_pages_mapping_new(wiki_pages)

    train_df = FH.pair_with_wiki_sentences(
        wiki_pages,
        datapack.train,
        mapping=wiki_mapping,
    )
    counts = train_df["label"].value_counts()
    print("Now using the following train data with 0 (Negative) and 1 (Positive)")
    print(counts)

    dev_evidences = FH.pair_with_wiki_sentences_eval(
        wiki_pages,
        datapack.dev,
        mapping=wiki_mapping,
    )
    dev_gt = JSONLineReader().read(dev_path)

    # Set up the TensorBoard writer
    writer = SummaryWriter(log_dir)
    print("Finished loading wiki_pages.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set up torch Datasets.
    train_dataset = SentRetrievalBERTDataset(train_df, tokenizer)
    val_dataset = SentRetrievalBERTDataset(dev_evidences, tokenizer)

    # Set up torch DataLoaders.
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=train_batch_size,
    )
    eval_dataloader = DataLoader(val_dataset, batch_size=test_batch_size)

    # Set up our model.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)

    if do_test_only:
        assert special_ckpt_name is not None
        assert special_ckpt_dir is not None
        assert test_set is not None
        assert pretrained_model_name is not None
        datapack.test = pd.DataFrame(test_set)

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name
        )
        model.to(device)
        model = load_model(model, special_ckpt_dir, special_ckpt_name)
        print("Do test only. Skip training!")
    else:
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        num_training_steps = num_epochs * len(train_dataloader)
        lr_scheduler = set_lr_scheduler(optimizer, num_training_steps)

        # Get ready to training.
        progress_bar = tqdm(range(num_training_steps))
        current_steps = 0
        validation_f1s = {}

        for epoch in range(num_epochs):
            model.train()

            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                writer.add_scalar("training_loss", loss.item(), current_steps)

                y_pred = torch.argmax(outputs.logits, dim=1).tolist()

                y_true = batch["labels"].tolist()
                # print(f"batch train acc: {accuracy_score(y_true, y_pred)}")

                current_steps += 1

                if current_steps % validation_step == 0 and current_steps > 0:
                    do_eval(
                        model=model,
                        device=device,
                        dev_loader=eval_dataloader,
                        df_evidences=dev_evidences,
                        ground_truths=dev_gt,
                        top_n=top_n,
                        validation_history=validation_f1s,
                        current_steps=current_steps,
                        writer=writer,
                        ckpt_dir=ckpt_dir,
                    )

        print("Finished training!")
        do_eval(
            model=model,
            device=device,
            dev_loader=eval_dataloader,
            df_evidences=dev_evidences,
            ground_truths=dev_gt,
            top_n=top_n,
            validation_history=validation_f1s,
            current_steps=current_steps,
            writer=writer,
            ckpt_dir=ckpt_dir,
        )
        best_ckpt_number = max(validation_f1s, key=validation_f1s.get)
        # best_ckpt_number = 100
        print(f"Validation history (step: F1): {validation_f1s}")
        print(
            f"Using the best mode: {best_ckpt_number} with F1 score: {validation_f1s[best_ckpt_number]}"
        )
        model = load_model(model, ckpt_dir, best_ckpt_number)
        print("Start final evaluations and write prediction files.")

        del train_df
        train_evidences = FH.pair_with_wiki_sentences_eval(
            wiki_pages,
            datapack.train,
            mapping=wiki_mapping,
        )
        train_gt = JSONLineReader().read(train_path)
        train_set = SentRetrievalBERTDataset(train_evidences, tokenizer)
        train_dataloader = DataLoader(train_set, batch_size=test_batch_size)

        print("Start calculating training scores")
        probs = get_predicted_probs(model, train_dataloader, device)
        train_results = evaluate_retrieval(
            probs=probs,
            df_evidences=train_evidences,
            ground_truths=train_gt,
            top_n=top_n,
            save_name=f"{save_dir}/train_doc{doc_n}sent{top_n}.jsonl",
        )
        print(f"Training scores => {train_results}")

        print("Start validation")
        probs = get_predicted_probs(model, eval_dataloader, device)
        val_results = evaluate_retrieval(
            probs=probs,
            df_evidences=dev_evidences,
            ground_truths=dev_gt,
            top_n=top_n,
            save_name=f"{save_dir}/dev_doc{doc_n}sent{top_n}.jsonl",
        )
        print(f"Validation scores => {val_results}")

    if do_test or do_test_only:
        test_evidences = FH.pair_with_wiki_sentences_eval(
            wiki_pages,
            datapack.test,
            mapping=wiki_mapping,
            is_testset=True,
        )
        test_gt = JSONLineReader().read(test_path)
        test_set = SentRetrievalBERTDataset(test_evidences, tokenizer)
        test_dataloader = DataLoader(test_set, batch_size=test_batch_size)

        print("Start testing")
        probs = get_predicted_probs(model, test_dataloader, device)
        evaluate_retrieval(
            probs=probs,
            df_evidences=test_evidences,
            ground_truths=test_gt,
            top_n=top_n,
            cal_scores=False,
            save_name=f"{save_dir}/test_doc{doc_n}sent{top_n}.jsonl",
        )
