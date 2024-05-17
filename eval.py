import json
import time
import re
import json
import numpy as np
import os
from tqdm import tqdm
import random
import argparse
import tiktoken
import hydra
from omegaconf import DictConfig
from glob import glob
import pandas as pd
from data import read_jsonl, parse_option, parse_answer


def compute_accuracy(gt, pred_solutions):
    if type(pred_solutions) == list:
        pred_answers = []
        for pred_solution in pred_solutions:
            pred_answers.append(pred_solution.strip())
        pred_answer = most_frequent(pred_answers)
    else:
        pred_answer = pred_solutions.strip()
    if type(gt) == list:
        gt = gt[0]

    # print(pred_answer, gt)
    if pred_answer == gt:
        return 1
    else:
        return 0


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num


@hydra.main(version_base=None, config_path="conf", config_name="eval")
def main(config: DictConfig) -> None:
    # set seed
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    results = read_jsonl(config["result_path"])
    out_path = config["out_path"]
    # subject = config["subject"]
    correct_list = []
    avg_len = []
    for i, r in enumerate(results):
        acc = compute_accuracy(r["gold"], r["extract"][:5])
        correct_list.append(acc)
        # sum len of preds
        len_sum = 0
        for pred in r["pred"][:5]:
            len_sum += len(pred)
        avg_len.append(len_sum)
    # open the out_path
    # with open(out_path, "a") as f:
    #     print(f"{subject:30s}\t{np.mean(correct_list)}", file=f)
    print(f"{np.mean(correct_list)}")
    print(f"{np.mean(avg_len)}")


if __name__ == "__main__":
    main()
