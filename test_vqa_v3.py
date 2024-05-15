import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from lavis.models import load_model_and_preprocess
from lavis.datasets.datasets.deepfake_datasets import DeepfakeEvalDataset
from lavis.tasks.vqa import VQATask

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Predefined paths for datasets
    dataset_info = [
        # {"vis_root": "/storage1/ruby/LAVIS/deepfake/ann/hive-test.json", "ann_path": "/storage1/ruby/LAVIS/deepfake/ann/hive-test.json", "output_json": "/storage1/ruby/LAVIS/lavis/output/BLIP2/dd-vqa/20240503172-instructBLIP_llm_lora/hive.json"},
        # {"vis_root": "/storage1/ruby/LAVIS/deepfake/ann/iDiff-test.json", "ann_path": "/storage1/ruby/LAVIS/deepfake/ann/iDiff-test.json", "output_json": "/storage1/ruby/LAVIS/lavis/output/BLIP2/dd-vqa/20240503172-instructBLIP_llm_lora/iDiff.json"},
        {"vis_root": "/storage1/ruby/LAVIS/deepfake/ann/mfg-test.json", "ann_path": "/storage1/ruby/LAVIS/deepfake/ann/mfg-test.json", "output_json": "/storage1/ruby/LAVIS/lavis/output/BLIP2/dd-vqa/20240503172-instructBLIP_llm_lora/mfg.json"},
        # {"vis_root": "/storage1/ruby/LAVIS/deepfake/ann/real-test.json", "ann_path": "/storage1/ruby/LAVIS/deepfake/ann/real-test.json", "output_json": "/storage1/ruby/LAVIS/lavis/output/BLIP2/dd-vqa/20240503172-instructBLIP_llm_lora/real.json"},
        # {"vis_root": "/storage1/ruby/LAVIS/deepfake/ann/sbi-test.json", "ann_path": "/storage1/ruby/LAVIS/deepfake/ann/sbi-test.json", "output_json": "/storage1/ruby/LAVIS/lavis/output/BLIP2/dd-vqa/20240503172-instructBLIP_llm_lora/sbi.json"},
        # {"vis_root": "/storage1/ruby/LAVIS/deepfake/ann/sladd-test.json", "ann_path": "/storage1/ruby/LAVIS/deepfake/ann/sladd-test.json", "output_json": "/storage1/ruby/LAVIS/lavis/output/BLIP2/dd-vqa/20240503172-instructBLIP_llm_lora/sladd.json"}
    ]

    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device
    )

    for dataset in dataset_info:
        vis_root = dataset["vis_root"]
        ann_path = dataset["ann_path"]
        output_json = dataset["output_json"]

        dataset = DeepfakeEvalDataset(
            vis_processors["eval"], txt_processors["eval"], vis_root, [ann_path]
        )
        dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

        vqa_task = VQATask(
            num_beams=3,
            max_len=50,
            min_len=1,
            evaluate=True,
            num_ans_candidates=128,
            inference_method="rank",
            prompt="Is this image real? If not, why?",
        )

        results = []
        for item in tqdm(dataloader):
            image = item["image"].to(device)
            text_input = item["text_input"]
            samples = {
                "image": image,
                "text_input": text_input,
                "question_id": item["question_id"],
                # "label": item["label"],
            }

            pred_qa_pairs = vqa_task.valid_step(model, samples)
            results.append(pred_qa_pairs[0])

        with open(output_json, "w") as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
