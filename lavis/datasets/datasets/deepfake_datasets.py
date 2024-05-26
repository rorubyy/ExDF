import torch
from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image
import os
import json
from collections import OrderedDict
import random


class DeepfakeUtils:
    @staticmethod
    def generate_instruction():
        question_templates = [
            "Is this image real or fake, and if fake, which features have been manipulated?",
            "Can you identify any alterations in this image?",
            "What makes this image real or fake?"
        ]
        probabilities = [0.9, 0.05, 0.05]
        selected_question = random.choices(question_templates, probabilities)[0]
        return selected_question


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
        self.positives = {"text": [], "images": []}
        self.negatives = {"text": [], "images": []}
        self.prepare_examples()

        
    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        instruction = DeepfakeUtils.generate_instruction()

        text_input = self.text_processor(instruction)
        text_output = self.text_processor(ann["text_output"])

        positive_texts = [self.text_processor(text) for text in self.positives["text"]]
        positive_images = [self.vis_processor(Image.open(os.path.join(self.vis_root, img)).convert("RGB")) for img in self.positives["images"]]

        negative_texts = [self.text_processor(text) for text in self.negatives["text"]]
        negative_images = [self.vis_processor(Image.open(os.path.join(self.vis_root, img)).convert("RGB")) for img in self.negatives["images"]]

        weights = [1]

        return {
            "image": image,
            "text_input": text_input,
            "text_output": text_output,
            "positive_texts": positive_texts,
            "positive_images": positive_images,
            "negative_texts": negative_texts,
            "negative_images": negative_images,
            "weights": weights,
            "label": ann["label"],
        }
        
        
    def prepare_examples(self):
        for idx, ann in enumerate(self.annotation):
            current_attributes = set(ann["attribute"])

            for i, a in enumerate(self.annotation):
                if i != idx:
                    if set(a["attribute"]) == current_attributes:
                        self.positives["text"].append(a["text_output"])
                        self.positives["images"].append(a["image"])
                        break

            for i, a in enumerate(self.annotation):
                if i != idx:
                    if set(a["attribute"]) != current_attributes:
                        self.negatives["text"].append(a["text_output"])
                        self.negatives["images"].append(a["image"])
                        break


class DeepfakeEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        self.vis_root = vis_root
        annotations = []
        for path in ann_paths:
            with open(path, "r") as file:
                annotations.extend(json.load(file))
        self.annotation = annotations
        # self.annotation = json.load(open(ann_paths[0]))

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        instruction = "Is this image real or fake, and if fake, which features have been manipulated?"
        text_input = self.text_processor(instruction)
        text_output = self.text_processor(ann["text_output"])

        return {
            "image": image,
            "text_input": text_input,
            "text_output": text_output,
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
            "label": ann["label"],
        }
