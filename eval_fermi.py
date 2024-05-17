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


def compute_accuracy(correct):
    if correct == "True":
        return 1
    else:
        return 0


@hydra.main(version_base=None, config_path="conf", config_name="eval")
def main(config: DictConfig) -> None:
    # set seed
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    results = read_jsonl(config["result_path"])
    out_path = config["out_path"]
    correct_list = []
    avg_len = []
    for i, r in enumerate(results):
        acc = compute_accuracy(r["correct"][0])
        correct_list.append(acc)
        avg_len.append(len(r["pred"][0]))
    print(f"{np.mean(correct_list)}")
    print(f"{np.mean(avg_len)}")


if __name__ == "__main__":
    main()
