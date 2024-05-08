import json
from sklearn.metrics import precision_recall_curve, auc


def simplify_answer(answer):
    return 1 if answer.lower().startswith("no") else 0


def load_data(file_path, increment_id=False):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def calculate_acc(gts, res):
    common_ids = set(gts.keys()) & set(res.keys())

    correct_predictions = sum(gts[qid] == res[qid] for qid in common_ids)
    accuracy = correct_predictions / len(common_ids)
    print("accuaracy:", accuracy)


def calculate_ap(gts, res):
    y_true = [gts[qid] for qid in sorted(gts)]
    y_scores = [res[qid] for qid in sorted(res)]
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = auc(recall, precision)
    print(f"Average Precision: {ap}")


def main():
    gt_files = [
        "/storage1/ruby/thesis_dataset/ann/test/classification/fixed_text_input/ip2p-test.json",
        "/storage1/ruby/LAVIS/deepfake/ann/real-test.json"
    ]
    ans_files = [
        "/storage1/ruby/LAVIS/result/classification/instructBLIP.json",
        "/storage1/ruby/LAVIS/lavis/output/BLIP2/dd-vqa/20240501150/real.json"
    ]

    gts, res = {}, {}

    data1 = load_data(gt_files[0])
    data2 = load_data(ans_files[0])
    for item in data1:
        gts[item["question_id"]] = [item["text_output"]]
    for item in data2:
        res[item["question_id"]] = [item["answer"]]


    for item in data1:
        gts[item["question_id"]] = simplify_answer(item["text_output"])
    for item in data2:
        res[item["question_id"]] = simplify_answer(item["answer"])
    calculate_acc(gts, res)


if __name__ == "__main__":
    main()
