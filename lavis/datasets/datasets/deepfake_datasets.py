import torch
from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image
import os
import json
from collections import OrderedDict
import random



class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "answers": "; ".join(ann["answer"]),
                "image": sample["image"],
            }
        )

class DeepfakeDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.positives = {}
        self.negatives = {}
        self.prepare_examples()

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        text_input = self.text_processor(ann["text_input"])
        text_output = self.text_processor(ann["text_output"])
        positive_outputs = [self.text_processor(pos["text_output"]) for pos in self.positives[index]]
        negative_outputs = [self.text_processor(neg["text_output"]) for neg in self.negatives[index]]


        weights = [1]  

        return {
            "image": image,
            "text_input": text_input,
            "text_output": text_output,
            "positive_outputs": positive_outputs,
            "negative_outputs": negative_outputs,
            "weights": weights,
        }


    def prepare_examples(self):
        for idx, ann in enumerate(self.annotation):
            current_attributes = set(ann["attribute"])
            positive_candidates = []
            negative_candidates = []

            for i, a in enumerate(self.annotation):
                if i != idx:
                    if set(a["attribute"]) == current_attributes:
                        positive_candidates.append(a)
                        if len(positive_candidates) == 3: 
                            break

            for i, a in enumerate(self.annotation):
                if i != idx:
                    if set(a["attribute"]) != current_attributes:
                        negative_candidates.append(a)
                        if len(negative_candidates) == 3:  # 当找到3个反例时停止搜索
                            break

            self.positives[idx] = positive_candidates if len(positive_candidates) == 3 else [ann]
            self.negatives[idx] = negative_candidates if len(negative_candidates) == 3 else [{"text_output": "No modifications detected."}]


class DeepfakeEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        self.vis_root = vis_root
        annotations = []
        for path in ann_paths:
            with open(path, 'r') as file:
                annotations.extend(json.load(file))
        self.annotation =  annotations
        # self.annotation = json.load(open(ann_paths[0]))

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        text_input = self.text_processor(ann["text_input"])
        text_output = self.text_processor(ann["text_output"])


        return {
            "image": image,
            "text_input": text_input,
            "text_output": text_output,
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
        }