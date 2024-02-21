import json
from pathlib import Path
import pandas as pd


def load_json_file(file_path):
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f.read().splitlines()]
    return data


def save_json_file(data: list, file_path: str):
    with open(file_path, "w") as f:
        for d in data:
            json.dump(d, f, ensure_ascii=False)
            f.write("\n")


def get_correct_evi(data: list, sent5_data: list):
    sent5_df = pd.DataFrame(sent5_data)

    for d in data:
        if d["label"] != "NOT ENOUGH INFO":
            gold_evidence = [evi[2:4] for eviset in d["evidence"] for evi in eviset]
            d["predicted_evidence"] = gold_evidence
        else:
            doc_id = d["id"]
            predicted_for_NE = sent5_df[sent5_df["id"] == doc_id][
                "predicted_evidence"
            ].values[0]
            d["predicted_evidence"] = predicted_for_NE

    return data


def make_correct_stage2(
    CFEVER_data_dir: str,
    doc_exp_gold_sent_dir: str,
    sent_exp_name: str,
    doc_n: int,
    sent_n: int,
) -> None:
    doc_exp_dir = doc_exp_gold_sent_dir.replace("_gold_sent", "")
    sent_exp_dir = f"{doc_exp_dir}/{sent_exp_name}"
    save_dir = f"{doc_exp_gold_sent_dir}/{sent_exp_name}"
    Path(save_dir).mkdir(parents=True, exist_ok=False)

    for d_type in ["train", "dev", "test"]:
        target_data = load_json_file(f"{CFEVER_data_dir}/{d_type}.jsonl")
        assert Path(f"{sent_exp_dir}/{d_type}_doc{doc_n}sent{sent_n}.jsonl").exists()

        source_sent = load_json_file(
            f"{sent_exp_dir}/{d_type}_doc{doc_n}sent{sent_n}.jsonl"
        )
        target_data = get_correct_evi(target_data, sent5_data=source_sent)
        save_json_file(target_data, f"{save_dir}/{d_type}_doc{doc_n}sent{sent_n}.jsonl")
