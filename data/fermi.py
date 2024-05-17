import json
import pandas as pd
from glob import glob
import re
import numpy as np
import argparse
from datasets import load_dataset


def reconst_instance(d):
    question = d["question"]
    answer = str(d["answer"])

    options = {}

    context = d["context"]
    contexts = re.split(r"=F\d:", context)[1:]
    contexts = [c.strip() for c in contexts]
    return {
        "question": question,
        "answer": answer,
        "options": options,
        "contexts": contexts,
    }


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--data_path",
        type=str,
        default="/home/juny116/Workspace/fermi/data/realFP/train_realfp.json",
    )
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--out_file", type=str, default="fermi_train.jsonl")
    args.add_argument("--max_examples", type=int, default=-1)
    args = args.parse_args()

    np.random.seed(args.seed)
    # load hf dataset
    # json load
    data = json.load(open(args.data_path))
    # data = load_dataset(args.data_path, split="test")

    if args.max_examples > 0:
        # np ramdom select max examples
        indices = np.random.choice(range(len(data)), args.max_examples, replace=False)
        # np int to int
        indices = [int(i) for i in indices]
        data = [data[i] for i in indices]

    new_data = []

    for d in data:
        new_data.append(reconst_instance(d))

    # # dump data as jsonl
    with open(args.out_file, "w") as f:
        for d in new_data:
            json.dump(d, f)
            f.write("\n")
