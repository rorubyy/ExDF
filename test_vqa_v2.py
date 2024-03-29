from lavis.datasets.datasets.deepfake_datasets import DeepfakeEvalDataset
from lavis.tasks.vqa import VQATask
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import random
import argparse
from PIL import Image
from tqdm import tqdm
import torch
from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.models import load_model_and_preprocess
import json

random.seed(43)

EXT = [".jpg", ".jpeg", ".png", ".JPEG"]


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vis_root",
        type=str,
        default="/storage1/ruby/LAVIS/deepfake/annotations/test/ffhq-real.json",
        help="The path to the image directory.",
    )
    parser.add_argument(
        "--ann_paths",
        type=str,
        default="/storage1/ruby/LAVIS/deepfake/annotations/test/ffhq-real.json",
        help="The path to the annotation directory.",
    )
    parser.add_argument(
        "--log", type=str, default="log/log.txt", help="Path to the log file."
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="/storage1/ruby/LAVIS/lavis/output/BLIP2/dd-vqa/sbi/real.json",
        help="The path to the output json file.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_name", type=str, default="blip2_vicuna_instruct")
    parser.add_argument("--model_type", type=str, default="vicuna7b")

    return parser.parse_args()


def main():

    args = arg_parser()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name=args.model_name, model_type=args.model_type, is_eval=True, device=device
    )

    dataset = DeepfakeEvalDataset(
        vis_processors["eval"], txt_processors["eval"], args.vis_root, [args.ann_paths]
    )
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    vqa_task = VQATask(
        num_beams=3,
        max_len=50,
        min_len=1,
        evaluate=True,
        num_ans_candidates=128,
        inference_method="rank",
        prompt="Is this photo real? If not, why?",
    )

    results = []
    for item in tqdm(dataloader):
        image = item["image"].to(device)
        text_input = item["text_input"]
        samples = {
            "image": image,
            "text_input": text_input,
            "question_id": item["question_id"],
        }

        pred_qa_pairs = vqa_task.valid_step(model, samples)
        print(pred_qa_pairs)
        if isinstance(pred_qa_pairs, list) and all(isinstance(item, dict) for item in pred_qa_pairs):
            results.extend(pred_qa_pairs) 
        else:
            results.append(pred_qa_pairs)
        
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
