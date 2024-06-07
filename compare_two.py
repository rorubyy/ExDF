import json

def load_json(path):
    with open(path, "r") as file:
        return json.load(file)
    
def compute_accuracy(gt_file_path, res_file_path):
    gt_data = load_json(gt_file_path)
    res_data = load_json(res_file_path)
    dataset_correct_counts = {}
    dataset_total_counts = {}
    for item in gt_data:
        question_id = item['question_id']
        gtAns = item["text_output"]
        dataset = item["dataset"]
        for item in res_data:
            if item["question_id"] == question_id:
                resAns = item["answer"]
        if dataset not in dataset_correct_counts:
            dataset_correct_counts[dataset]=0
            dataset_total_counts[dataset]=0
        dataset_total_counts[dataset] += 1
        if ("no" in resAns and "No" in gtAns) or ("yes" in resAns and "Yes" in gtAns):
            dataset_correct_counts[dataset] += 1
            
    for dataset in dataset_total_counts:
        if dataset == "real":
            continue
        real_correct_count = dataset_correct_counts.get("real", 0)
        real_total_count = dataset_total_counts.get("real", 0)
        dataset_correct_count = dataset_correct_counts[dataset]
        dataset_total_count = dataset_total_counts[dataset]

        combined_correct_count = dataset_correct_count + real_correct_count
        combined_total_count = dataset_total_count + real_total_count

        accuracy = combined_correct_count / combined_total_count if combined_total_count > 0 else 0
        print(f"Accuracy for {dataset} : {accuracy:.4f}")


# Example usage
gt_file_path = "/storage1/ruby/LAVIS/ExDF/ann/test/test-ip2p.json"
res_file_path = "/storage1/ruby/LAVIS/lavis/output/ip2p_mask.json"
accuracy_metrics = compute_accuracy(gt_file_path, res_file_path)
