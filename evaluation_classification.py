import json
from sklearn.metrics import precision_recall_curve, auc


def simplify_answer(answer):
    return 1 if "fake" in answer.lower()  or "no" in answer.lower() else 0


def load_data(file_path, increment_id=False):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def calculate_metrics(gts, res):
    common_ids = set(gts.keys()) & set(res.keys())
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    correct_predictions = 0
    for qid in common_ids:
        if gts[qid] == res[qid]:
            correct_predictions += 1
            if gts[qid] == 1:  
                true_positives += 1
        else:
            if res[qid] == 1:
                false_positives += 1
            if gts[qid] == 1:  
                false_negatives += 1

    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 1.0 

    if true_positives + false_negatives > 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = 1.0  

    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    accuracy = correct_predictions / len(common_ids)

    print(f"Accuracy (ACC): {accuracy:.5f}")
    print(f"Precision (AP): {precision:.5f}")
    print(f"Recall (R_ACC): {recall:.5f}")
    print(f"F1 Score (F_ACC): {f1_score:.5f}")


def main():
    gt_files = "/storage1/ruby/thesis_dataset/ann/ff++/FF++.json"
    ans_files = "/storage1/ruby/LAVIS/lavis/output/BLIP2/ff++/20240523145/result/test_vqa_result_rank0.json"

    gts, res = {}, {}

    data1 = load_data(gt_files)
    data2 = load_data(ans_files)
    for item in data1:
        gts[item["question_id"]] = simplify_answer(item["text_output"])
    for item in data2:
        res[item["question_id"]] = simplify_answer(item["answer"])
    calculate_metrics(gts, res)


if __name__ == "__main__":
    main()
