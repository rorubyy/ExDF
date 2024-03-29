import json
from collections import defaultdict
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from bert_score import score as bert_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score


def bleu(gts, res):
    scorer = Bleu(n=4)
    score, scores = scorer.compute_score(gts, res)

    print("bleu = %s" % score)


def cider(gts, res):
    scorer = Cider()
    (score, scores) = scorer.compute_score(gts, res)
    print("cider = %s" % score)


def meteor(gts, res):
    scorer = Meteor()
    score, scores = scorer.compute_score(gts, res)
    print("meter = %s" % score)


def rouge(gts, res):
    scorer = Rouge()
    score, scores = scorer.compute_score(gts, res)
    print("rouge = %s" % score)


def spice(gts, res):
    scorer = Spice()
    score, scores = scorer.compute_score(gts, res)
    print("spice = %s" % score)


def bertScore(gts, res, common_ids):
    print("Computing BERT Score...")
    all_references = [gts[qid] for qid in common_ids]
    all_hypotheses = [res[qid] for qid in common_ids]
    P, R, F1 = bert_score(
        all_hypotheses, all_references, lang="en", rescale_with_baseline=True
    )
    print(
        f"BERT Score: Precision: {P.mean().item()}, Recall: {R.mean().item()}, F1: {F1.mean().item()}"
    )


def comupte_individual_dataset(data1, data2, common_ids, res, gts):
    dataset_gts = defaultdict(dict)
    dataset_res = defaultdict(dict)
    dataset_simplified_gts = defaultdict(dict)
    dataset_simplified_res = defaultdict(dict)

    for item in data1:
        if "text_output" in item and item["question_id"] in common_ids:
            dataset_name = item["image"].split("/")[-2]  # 根据路径提取数据集名称
            dataset_gts[dataset_name][item["question_id"]] = [item["text_output"]]
            dataset_simplified_gts[dataset_name][item["question_id"]] = simplify_answer(item["text_output"])

    for item in data2:
        if "answer" in item and item["question_id"] in common_ids:
            for dataset_name in dataset_gts:
                if item["question_id"] in dataset_gts[dataset_name]:
                    dataset_res[dataset_name][item["question_id"]] = [item["answer"]]
                    dataset_simplified_res[dataset_name][item["question_id"]] = simplify_answer(item["answer"])    
    # for qid, answer in res.items():
    #     for dataset_name in dataset_gts:
    #         if qid in dataset_gts[dataset_name]:
    #             dataset_res[dataset_name][qid] = answer
    #             dataset_simplified_res[dataset_name][item["question_id"]] = simplify_answer(answer[0])


    for dataset_name in dataset_res:
        print(f"Scores for dataset {dataset_name}:")
        bleu(dataset_gts[dataset_name], dataset_res[dataset_name])
        cider(dataset_gts[dataset_name], dataset_res[dataset_name])
        meteor(dataset_gts[dataset_name], dataset_res[dataset_name])
        rouge(dataset_gts[dataset_name], dataset_res[dataset_name])
        spice(dataset_gts[dataset_name], dataset_res[dataset_name])
        refs = [gts[qid][0] for qid in dataset_gts[dataset_name]]
        hyps = [res[qid][0] for qid in dataset_res[dataset_name]]
        P, R, F1 = bert_score(hyps, refs, lang='en', rescale_with_baseline=True)
        print(f"BERTScore for {dataset_name} - Precision: {P.mean().item()}, Recall: {R.mean().item()}, F1: {F1.mean().item()}")
        correct_predictions = sum(
            dataset_simplified_gts[dataset_name][qid] == dataset_simplified_res[dataset_name][qid]
            for qid in dataset_gts[dataset_name]
        )
        accuracy = correct_predictions / len(dataset_gts[dataset_name])
        print(f"Accuracy for {dataset_name}: {accuracy}")

def cosine_similarity_scores(gts, res):
    gts_texts = [gts[qid][0] for qid in gts]
    res_texts = [res[qid][0] for qid in res]
    length = len(gts)
    sim = 0

    vectorizer = TfidfVectorizer()
    all_texts = gts_texts + res_texts
    all_vectors = vectorizer.fit_transform(all_texts)

    for i, qid in enumerate(gts):
        gts_vector = all_vectors[i : i + 1]
        res_vector = all_vectors[len(gts) + i : len(gts) + i + 1]
        sim += cosine_similarity(gts_vector, res_vector)[0][0]
        # print(f"Cosine similarity for question_id {qid}: {sim}")
    print("Cosine similarity", sim / length)


def calculate_bertscore(gts, res, common_ids):
    refs = [gts[qid][0] for qid in common_ids]
    hyps = [res[qid][0] for qid in common_ids]

    P, R, F1 = score(hyps, refs, lang="en", verbose=True)

    print(f"BERTScore Precision: {P.mean().item()}")
    print(f"BERTScore Recall: {R.mean().item()}")
    print(f"BERTScore F1: {F1.mean().item()}")


def simplify_answer(answer):
    fake = "The photo is fake"
    return 1 if fake.lower() in answer.lower() else 0


def main():
    ans_file = "/storage1/ruby/AntifakePrompt/answers.json"
    gt_file = "/storage1/ruby/instruct-pix2pix/data/ffhq-mini-test.json"

    with open(gt_file, "r") as f:
        data1 = json.load(f)

    with open(ans_file, "r") as f:
        data2 = json.load(f)

    gts = {}
    res = {}

    for item in data1:
        gts[item["question_id"]] = [item["text_output"]]

    for item in data2:
        res[item["question_id"]] = [item["answer"]]

    common_ids = set(gts.keys()) & set(res.keys())
    gts = {qid: gts[qid] for qid in common_ids}
    res = {qid: res[qid] for qid in common_ids}

    # cosine_similarity_scores(gts, res)
    # calculate_bertscore(gts, res, common_ids)
    
    # bleu(gts, res)
    # cider(gts, res)
    # meteor(gts, res)
    # rouge(gts, res)
    # spice(gts, res)
    # # compute f1 acc
    # for item in data1:
    #     gts[item["question_id"]] = simplify_answer(item["text_output"])

    # for item in data2:
    #     res[item["question_id"]] = simplify_answer(item["answer"])
    # correct_predictions = sum(gts[qid] == res[qid] for qid in common_ids)
    # accuracy = correct_predictions / len(common_ids)
    # print("accuaracy:", accuracy)


    comupte_individual_dataset(data1, data2, common_ids, res, gts)


if __name__ == "__main__":
    main()
