import json
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from bert_score import score as bert_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score


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
    gt_files = "/storage1/ruby/thesis_dataset/ann/test/generation/random_text_input/ip2p-test.json"
    ans_files = "/storage1/ruby/LAVIS/lavis/output/result/generation/multitask-random-classification-llminput.json"

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

import json
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from bert_score import score as bert_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score


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
    gt_files = "/storage1/ruby/thesis_dataset/ann/test/generation/random_text_input/iDiff-test.json"
    ans_files = "/storage1/ruby/LAVIS/lavis/output/result/generation/stage2_instructBLIP_random_text.json"

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
