import torch
from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image
import os
import json
from collections import OrderedDict
from torchvision import transforms
import cv2
import numpy as np
import dlib


def extract_landmarks(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("ExDF/shape_predictor_68_face_landmarks.dat")

    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)

    if len(rects) > 0:
        shape = predictor(gray, rects[0])
        landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        return landmarks
    else:
        return None


def generate_face_and_head_mask(image, landmarks):
    image_size = image.size[0]
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    face_polygon = landmarks_to_polygon(landmarks)
    cv2.fillPoly(mask, [np.array(face_polygon, dtype=np.int32)], 1)
    mask = cv2.dilate(mask, np.ones((50, 50), np.uint8), iterations=1)
    return mask


def landmarks_to_polygon(landmarks):
    return [(int(x), int(y)) for x, y in landmarks]


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

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        landmarks = extract_landmarks(image)
        if landmarks:
            face_head_mask = generate_face_and_head_mask(image, landmarks)
            face_head_mask = torch.tensor(face_head_mask, dtype=torch.float32)
            face_head_mask = torch.nn.functional.interpolate(
                face_head_mask.unsqueeze(0).unsqueeze(0),
                size=(224, 224),
                mode="bilinear",
            ).squeeze()
        else:
            face_head_mask = torch.ones((224, 224), dtype=torch.float32)

        image = self.vis_processor(image)
        text_input = self.text_processor(ann["text_input"])
        text_output = self.text_processor(ann["text_output"])

        weights = 0.1 if "all" in ann["attribute"] else 1.0

        return {
            "image": image,
            "text_input": text_input,
            "text_output": text_output,
            "weights": weights,
            "mask": face_head_mask.unsqueeze(0),
        }


class DeepfakeEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        self.vis_root = vis_root
        self.annotation = json.load(open(ann_paths[0]))
        self.mask_to_tensor = transforms.ToTensor()

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        
        landmarks = extract_landmarks(image)
        if landmarks:
            face_head_mask = generate_face_and_head_mask(image, landmarks)
            face_head_mask = torch.tensor(face_head_mask, dtype=torch.float32)
            face_head_mask = torch.nn.functional.interpolate(
                face_head_mask.unsqueeze(0).unsqueeze(0),
                size=(224, 224),
                mode="bilinear",
            ).squeeze()
        else:
            face_head_mask = torch.ones((224, 224), dtype=torch.float32)


        image = self.vis_processor(image)
        text_input = self.text_processor(ann["text_input"])
        text_output = self.text_processor(ann["text_output"])

        return {
            "image": image,
            "text_input": text_input,
            "text_output": text_output,
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
            "mask": face_head_mask.unsqueeze(0),
            # "attribute": "; ".join(ann["attribute"])
        }
