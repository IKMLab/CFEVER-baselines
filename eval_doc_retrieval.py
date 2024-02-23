import argparse
import pandas as pd
from utils import (
    load_json,
    calculate_precision,
    calculate_recall,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_file", type=str, required=True)
    parser.add_argument("--doc_pred_file", type=str, required=True)
    parser.add_argument("--pred_colname", type=str, default="predicted_pages")
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--print_avg_num_pages", action="store_true")
    args = parser.parse_args()

    data = load_json(args.doc_pred_file)
    if args.pred_colname not in data[0]:
        test_data = load_json(args.source_file)
        pred_ids = [int(list(d.keys())[0]) for d in data]
        test_ids = [d["id"] for d in test_data]
        assert pred_ids == test_ids
        for d, pred in zip(test_data, data):
            predictions = list(set(list(pred.values())[0]))
            print("Claim: ", d["claim"])
            print(predictions)
            print("=" * 30)
            d[args.pred_colname] = predictions
        data = test_data

    if args.top_k > 0:
        for d in data:
            d[args.pred_colname] = d[args.pred_colname][: args.top_k]

    if args.print_avg_num_pages:
        num_pages = [len(d[args.pred_colname]) for d in data]
        print(f"Average number of pages: {sum(num_pages) / len(num_pages)}")
        print(num_pages)
        print(len(num_pages))

    data_df = pd.DataFrame(data)

    precision = calculate_precision(data, data_df[args.pred_colname])
    recall = calculate_recall(data, data_df[args.pred_colname])
    results = {
        "f1_score": 2.0 * precision * recall / (precision + recall),
        "precision": precision,
        "recall": recall,
    }
    for k, v in results.items():
        print(f"{k}: {v}")
