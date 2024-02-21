import argparse
from pathlib import Path
from claim_verification.bert_rte import claim_verification_main
from common.util.args_handler import save_args_to_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--sent_exp_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="hfl/chinese-bert-wwm-ext")
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--evidence_topk", type=int, default=5)
    parser.add_argument("--validation_step", type=int, default=25)
    parser.add_argument("--output_filename", type=str, default=f"submission.jsonl")
    args = parser.parse_args()

    rte_exp_name = "rte_" + args.model_name.split("/")[-1]

    save_dir = f"{args.data_dir}/{args.sent_exp_name}/{rte_exp_name}"
    Path(save_dir).mkdir(parents=False, exist_ok=False)
    save_args_to_json(args, f"{save_dir}/args.json")

    claim_verification_main(
        data_dir=args.data_dir,
        save_dir=save_dir,
        sent_exp_name=args.sent_exp_name,
        rte_exp_name=rte_exp_name,
        model_name=args.model_name,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        seed=args.seed,
        learning_rate=args.lr,
        num_epochs=args.num_epochs,
        max_seq_len=args.max_seq_len,
        evidence_topk=args.evidence_topk,
        validation_step=args.validation_step,
        output_filename=args.output_filename,
    )
