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
from langchain_google_genai import GoogleGenerativeAI
import google.generativeai as genai
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
)
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai as genai

GOOGLE_API_KEY = "AIzaSyCc-F_Bq_k3HFbAv3ZjA6xd5e-AeYyds3E"
GOOGLE_API_KEY = "AIzaSyDsNzzaXawDm1C-Aq5SQCJKi4KkNBeCwXc"
genai.configure(api_key=GOOGLE_API_KEY)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    # set seed
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    max_examples = config["max_examples"]
    # load model
    model_config = config["model"]
    # if model_config["model_type"] == "llm":
    #     model = OpenAI(
    #         model_name=model_config["model_name_or_path"],
    #         temperature=model_config["temperature"],
    #         max_tokens=model_config["max_tokens"],
    #         openai_api_base=model_config["openai_api_base"],
    #         openai_organization=model_config["openai_organization"],
    #         request_timeout=60,
    #     )
    # elif model_config["model_type"] == "chat":
    #     model = ChatOpenAI(
    #         model_name=model_config["model_name_or_path"],
    #         temperature=model_config["temperature"],
    #         max_tokens=model_config["max_tokens"],
    #         openai_api_base=model_config["openai_api_base"],
    #         openai_organization=model_config["openai_organization"],
    #         request_timeout=60,
    #     )
    # model = GoogleGenerativeAI(
    #     model="gemini-pro",
    #     google_api_key=GOOGLE_API_KEY,
    #     max_tokens=model_config["max_tokens"],
    #     temperature=model_config["temperature"],
    # )
    # model = ChatGoogleGenerativeAI(
    #     model="gemini-pro",
    #     google_api_key=GOOGLE_API_KEY,
    #     safety_settings={
    #         HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    #         HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    #         HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    #         HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    #     },
    # )
    generation_config = genai.types.GenerationConfig(
        # Only one candidate for now.
        candidate_count=1,
        max_output_tokens=model_config["max_tokens"],
        temperature=model_config["temperature"],
    )
    model = genai.GenerativeModel(
        "gemini-pro",
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        },
    )

    instruction = config["template"]["instruction"]
    form = config["template"]["format"]
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
        # if options are provided, use them
        options = ""
        for option, answer in d["options"].items():
            options += f"{option}. {answer}\n"
        # if facts are provided, use them
        context = ""
        if "contexts" in d:
            for k, c in enumerate(d["contexts"]):
                context += f"Fact {k+1}. {c}\n"
            context += "\n"
        query = (
            form.replace("[QUESTION]", d["question"])
            .replace("[OPTIONS]", options)
            .replace("[ANSWER]", "")
            .replace("[CONTEXTS]", context)
        )
        context = prompt + query
        input_prompt = context
        if i == 0:
            print(input_prompt)
            print("=====================================")
        for r in range(0, config["rounds"]):
            # set 30 second timeout
            max_retries = 4
            delay = 2
            retry_count = 0
            while retry_count < max_retries:
                try:
                    result = model.generate_content(context)
                    print(result.prompt_feedback)
                    answers.append(result.text)
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    time.sleep(delay)
                    retry_count += 1

            # print(result)
            # if model_config["model_type"] == "chat":
            #     result = result.content

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
