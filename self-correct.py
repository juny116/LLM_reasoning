import json
import numpy as np
import random
# from langchain.chat_models import ChatOpenAI
# from langchain.llms import OpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from tqdm import tqdm
import random
import hydra
from omegaconf import DictConfig
from data import read_jsonl, parse_answer
from utils import messages_to_string
import os
import time


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    # set seed
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    max_examples = config["max_examples"]
    # load model
    model_config = config["model"]
    if model_config["model_type"] == "llm":
        model = OpenAI(
            model_name=model_config["model_name_or_path"],
            temperature=model_config["temperature"],
            max_tokens=model_config["max_tokens"],
            openai_api_base=model_config["openai_api_base"],
            request_timeout=60,
        )
    elif model_config["model_type"] == "chat":
        model = ChatOpenAI(
            model_name=model_config["model_name_or_path"],
            temperature=model_config["temperature"],
            max_tokens=model_config["max_tokens"],
            openai_api_base=model_config["openai_api_base"],
            request_timeout=60,
        )

    instruction = config["template"]["instruction"]
    form = config["template"]["format"]
    revise_inst = config["template"]["revise"]
    feedback_inst = config["template"]["feedback"]

    if config["train_path"]:
        train_data = read_jsonl(config["train_path"])
    else:
        train_data = []
    test_data = read_jsonl(config["test_path"])
    prompt = instruction
    for d in train_data:
        options = ""
        for option, answer in d["options"].items():
            options += f"{option}. {answer}\n"
        prompt += (
            form.replace("[QUESTION]", d["question"])
            .replace("[OPTIONS]", options)
            .replace("[ANSWER]", f' {d["answer"]}')
        )
        prompt += "\n\n"

    # if the result file exists, read it
    if os.path.exists(config["out_path"]):
        results = read_jsonl(config["out_path"])
    else:
        results = []
    # if len(results) > 0 continue from last checkpoint
    for i, d in enumerate(tqdm(test_data[:max_examples])):
        if len(results) > 0 and i < len(results):
            continue
        answers = []
        feedbacks = []
        options = ""
        for option, answer in d["options"].items():
            options += f"{option}. {answer}\n"
        query = (
            form.replace("[QUESTION]", d["question"])
            .replace("[OPTIONS]", options)
            .replace("[ANSWER]", "")
        )
        context = prompt + query
        if model_config["model_type"] == "chat":
            context = [
                HumanMessage(content=context),
            ]
            input_prompt = messages_to_string(context)
        else:
            input_prompt = context
        if i == 0:
            print(input_prompt)
        # Init answer
        result = model.invoke(context)
        if model_config["model_type"] == "chat":
            result = result.content
        answers.append(result)
        if i == 0:
            print(result)
        for r in range(0, config["rounds"]):
            # set 30 second timeout
            query = (
                form.replace("[QUESTION]", d["question"])
                .replace("[OPTIONS]", options)
                .replace("[ANSWER]", f"{result}")
            )
            context = prompt + query + "\n\n" + feedback_inst
            if model_config["model_type"] == "chat":
                context = [
                    HumanMessage(content=context),
                ]
                input_prompt = messages_to_string(context)
            else:
                input_prompt = context
            # feedback
            feedback_result = model(context)
            if model_config["model_type"] == "chat":
                feedback_result = feedback_result.content
            feedbacks.append(feedback_result)
            if i == 0:
                print(input_prompt)
                print(feedback_result)
            if "[ACCURATE]" in feedback_result:
                break

            # revise
            revise_query = revise_inst.replace("[FEEDBACK]", feedback_result)
            context = prompt + query + "\n\n" + revise_query
            if model_config["model_type"] == "chat":
                context = [
                    HumanMessage(content=context),
                ]
                input_prompt = messages_to_string(context)
            else:
                input_prompt = context
            result = model(context)
            if model_config["model_type"] == "chat":
                result = result.content
            answers.append(result)
            if i == 0:
                print(input_prompt)
                print(result)
                print("=====================================")

        results.append(
            {
                "question": d["question"],
                "input": input_prompt,
                "pred": answers,
                "gold": d["answer"],
                "options": d["options"],
            }
        )

        # write results to jsonl
        out_path = config["out_path"]
        with open(out_path, "w") as f:
            for d in results:
                json.dump(d, f)
                f.write("\n")

        if model_config["name"] == "chatGPT":
            time.sleep(2)


if __name__ == "__main__":
    main()
