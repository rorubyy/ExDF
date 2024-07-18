import json
from pycocoevalcap.spice.spice import Spice

def load_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def spice(gts, res):
    scorer = Spice()
    score, scores = scorer.compute_score(gts, res)
    print("SPICE = %s" % score)

def calculate_scores(fake_gts, real_gts, fake_res, real_res, dataset_name):
    gts = {**fake_gts, **real_gts}
    res = {**fake_res, **real_res}
    print(f"Calculating scores for dataset {dataset_name}")
    spice(gts, res)

def main():
    gt_files = "/home/u2272230/ExDF/ann/test/test-hive.json"
    ans_files = "/home/u2272230/deepfake_explanation/lavis/output/BLIP2/hive_mask_atts_v2/20240617160/result/test_vqa_result_rank0.json"

    gts_data = load_data(gt_files)
    res_data = load_data(ans_files)
    
    gts = {}
    res = {}

    for item in gts_data:
        question_id = item["question_id"]
        dataset = item["dataset"]
        if dataset not in gts:
            gts[dataset] = {}
        gts[dataset][question_id] = [item["text_output"]]

    for item in res_data:
        question_id = item["question_id"]
        for dataset in gts.keys():
            if question_id in gts[dataset]:
                if dataset not in res:
                    res[dataset] = {}
                res[dataset][question_id] = [item["answer"]]
                break

    real_gts = gts.get("real", {})
    real_res = res.get("real", {})

    valid_datasets = {"idiff", "ip2p", "hive", "mfg"}

    for dataset in gts.keys():
        if dataset not in valid_datasets:
            continue
        fake_gts = gts[dataset]
        fake_res = res.get(dataset, {})
        calculate_scores(fake_gts, real_gts, fake_res, real_res, dataset)

if __name__ == "__main__":
    main()
