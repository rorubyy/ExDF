import json
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.spice.spice import Spice


def load_data(file_path, increment_id=False):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def meteor(gts, res):
    scorer = Meteor()
    score, scores = scorer.compute_score(gts, res)
    print("meter = %s" % score)


def spice(gts, res):
    scorer = Spice()
    score, scores = scorer.compute_score(gts, res)
    print("spice = %s" % score)


def main():
    gt_files = "/storage1/ruby/thesis_dataset/ann/test/generation/fixed_text_input/mfg-test.json"
    ans_files = "/storage1/ruby/LAVIS/lavis/output/result/classification/test.json"

    gts, res = {}, {}

    data1 = load_data(gt_files)
    data2 = load_data(ans_files)
    question_ids1 = {item["question_id"] for item in data1}
    question_ids2 = {item["question_id"] for item in data2}
    common_ids = question_ids1 & question_ids2

    gts = {item["question_id"]: [item["text_output"]] for item in data1 if item["question_id"] in common_ids}
    res = {item["question_id"]: [item["answer"]] for item in data2 if item["question_id"] in common_ids}
    
    spice(gts, res)
    meteor(gts, res)


if __name__ == "__main__":
    main()
