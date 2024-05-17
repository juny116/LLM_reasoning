import json
import numpy as np
import random
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from tqdm import tqdm
import random
import hydra
from omegaconf import DictConfig
from data import read_jsonl, parse_answer
from utils import messages_to_string
import os
import time


@hydra.main(version_base=None, config_path="conf", config_name="extract")
def main(config: DictConfig) -> None:
    # set seed
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    max_examples = 9999
    # load model
    model_config = config["model"]
    if model_config["model_type"] == "llm":
        model = OpenAI(
            model_name=model_config["model_name_or_path"],
            temperature=model_config["temperature"],
            max_tokens=model_config["max_tokens"],
            openai_api_base=model_config["openai_api_base"],
            request_timeout=30,
        )
    elif model_config["model_type"] == "chat":
        model = ChatOpenAI(
            model_name=model_config["model_name_or_path"],
            temperature=model_config["temperature"],
            max_tokens=model_config["max_tokens"],
            openai_api_base=model_config["openai_api_base"],
            request_timeout=30,
        )
        # max_examples = 100
    instruction = config["template"]["instruction"]
    form = config["template"]["format"]
    train_data = read_jsonl(config["train_path"])
    test_data = read_jsonl(config["test_path"])
    prompt = instruction
    for d in train_data:
        prompt += form.replace("[SOLUTION]", d["pred"][0]).replace(
            "[ANSWER]", f' {d["extract"]}'
        )
        prompt += "\n\n"

    # if the result file exists, read it
    if os.path.exists(config["out_path"]):
        results = read_jsonl(config["out_path"])
    else:
        results = []
    # if len(results) > 0 continue from last checkpoint
    start = True
    for i, d in enumerate(tqdm(test_data[:max_examples])):
        if len(results) > 0 and i < len(results):
            continue
        answers = []
        for pred in d["pred"]:
            query = form.replace("[SOLUTION]", pred).replace("[ANSWER]", "")
            context = prompt + query
            if model_config["model_type"] == "chat":
                context = [
                    HumanMessage(content=context),
                ]
            if start:
                if model_config["model_type"] == "chat":
                    print(messages_to_string(context))
                else:
                    print(context)
                print("================ EOE =====================")
                start = False

            result = model.invoke(context)
            # print(result)
            if model_config["model_type"] == "chat":
                result = result.content
            result = result.split("\n")[0].replace("$", "").strip()
            # print("----------------- PRED --------------------")
            # print(pred)
            # print("----------------- EXTRACT --------------------")
            # print(result)
            # print("-------------------------------------")
            answers.append(result)

        results.append(
            {
                "question": d["question"],
                "pred": d["pred"],
                "gold": d["gold"],
                "options": d["options"],
                "extract": answers,
                "input": d["input"],
            }
        )

        # write results to jsonl
        out_path = config["out_path"]
        with open(out_path, "w") as f:
            for d in results:
                json.dump(d, f)
                f.write("\n")

        # if model_config["name"] == "chatGPT":
        #     time.sleep(2)


if __name__ == "__main__":
    main()
