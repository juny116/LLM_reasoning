# open self-correct1

from data import read_jsonl, parse_option, parse_answer
import numpy as np


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num


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


result_path = "results/gsm/chatGPT/extract/self-correct2.jsonl"
results = read_jsonl(result_path)

correct_list = []
cor_to_inc = []
cor_to_inc_len = []
inc_to_cor = []
inc_to_cor_len = []
len_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
for i, r in enumerate(results):
    len_dict[len(r["extract"])] += 1
    if len(r["extract"]) == 1:
        continue

    og = compute_accuracy(r["gold"], r["extract"][0])
    acc = compute_accuracy(r["gold"], r["extract"][:5][-1])
    if og == 1 and acc == 0:
        cor_to_inc.append(r)
        cor_to_inc_len.append(len(r["pred"][0]))
    elif og == 0 and acc == 1:
        inc_to_cor.append(r)
        inc_to_cor_len.append(len(r["pred"][0]))

    correct_list.append(acc)
    # acc = compute_accuracy(r["gold"], r["extract"][:5])
    # correct_list.append(acc)
    # # sum len of preds
    # len_sum = 0
    # for pred in r["pred"][:5]:
    #     len_sum += len(pred)
    # avg_len.append(len_sum)

print(len(cor_to_inc))
print(len(inc_to_cor))
print(len_dict)
print(f"{np.mean(correct_list)}")
print(f"{np.mean(cor_to_inc_len)}")
print(f"{np.mean(inc_to_cor_len)}")
