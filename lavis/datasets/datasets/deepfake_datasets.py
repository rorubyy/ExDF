import torch
from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image
import os
import json
from collections import OrderedDict



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
                "label": sample["label"],
            }
        )

class DeepfakeDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        text_input = self.text_processor(ann["text_input"])
        text_output = self.text_processor(ann["text_output"])

        weights = [1]  

        return {
            "image": image,
            "text_input": text_input,
            "text_output": text_output,
            "weights": weights,
            "label": ann["label"],
        }


class DeepfakeEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        self.vis_root = vis_root
        self.annotation = json.load(open(ann_paths[0]))

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
            "label": ann["label"],
        }