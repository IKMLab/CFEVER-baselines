import argparse
from pathlib import Path
from sentence_retrieval.bert_binary import sent_retrieval_main
from common.util.args_handler import save_args_to_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/bm25")
    parser.add_argument("--stage_1_doc_n", type=int, required=True)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--negative_ratio", type=float, default=0.032)
    parser.add_argument("--validation_step", type=int, default=100)
    parser.add_argument(
        "--top_n", type=int, default=5, help="Top n sentences to retrieve"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="hfl/chinese-bert-wwm-ext")
    args = parser.parse_args()

    sent_exp_name = "sent_" + args.model_name.split("/")[-1]
    save_dir = f"{args.data_dir}/{sent_exp_name}"
    Path(save_dir).mkdir(parents=False, exist_ok=False)
    save_args_to_json(args, f"{save_dir}/args.json")

    sent_retrieval_main(
        data_dir=args.data_dir,
        save_dir=save_dir,
        sent_exp_name=sent_exp_name,
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        negative_ratio=args.negative_ratio,
        validation_step=args.validation_step,
        seed=args.seed,
        doc_n=args.stage_1_doc_n,
        top_n=args.top_n,
    )
