import pandas as pd
import numpy as np
import os
import time
import random 
import argparse
from PIL import Image
from os import makedirs
from os.path import dirname
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
from torch.utils.data import DataLoader
from lavis.datasets.datasets.base_dataset import BaseDataset
import json

import lavis
from lavis.models import load_model_and_preprocess

random.seed(43)

EXT = ['.jpg', '.jpeg', '.png', '.JPEG']

class DDVQADataset(BaseDataset):
    
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, [ann_paths])


    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = ann["image"]
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image) 
        return {
            "image": image,
            "question_id": int(ann["question_id"]),
            "text_input" : self.text_processor(ann["text_input"]),
            "text_output" : self.text_processor(ann["text_output"]),
        }
    
class InstructBLIP():
    def __init__(self, name="instruct_vicuna7b", model_type="vicuna7b", is_eval=True, device="cpu") -> None:
        #self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(name, model_type, is_eval, device)
        self.imgs = []
        self.labels = []
        
        # QA
        self.question = ""
        
        # results
        self.acc = None
        self.confusion_mat = None
        
        self.acc_3class = None
        self.confusion_mat_3class = None
        
        self.com_acc = None
        self.com_confusion_mat = None
        self.uncom_acc = None
        self.uncom_confusion_mat = None

    def LoadModels(self, model, vis_processors, txt_processors, device):
        self.model = model
        self.vis_processors = vis_processors
        self.txt_processors = txt_processors
        self.device = device
        
    def LoadData(self, vis_root, ann_paths):
        self.dataset = DDVQADataset(self.vis_processors["eval"], self.txt_processors["eval"], vis_root, ann_paths)
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=1, shuffle=False, num_workers=8)    
    
    def QueryImgs_batch(self, question, true_string="yes", logPath='log.txt'):

        answers_list = []

        with torch.no_grad():
            for item in tqdm(self.dataloader):
                image = item['image'].to(self.device)
                question = item['text_input']
                samples = {"image": image, "text_input": question}
                question_id = item["question_id"]
                ans = self.model.predict_answers(samples=samples, inference_method="generate")
                print(ans)
                answers_list.append({"file_name": question_id.item(), "answer": ans})
                
        with open('answers.json', 'w') as json_file:
            json.dump(answers_list, json_file)

    
    def Query(self, image, question):
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        
        samples = {"image": image, "prompt": question}
        candidates = ["yes", "no"]
        ans = self.model.predict_class(samples=samples, candidates=candidates)
        pred_label = ["Real" if candidates[list(a).index(0)]=="yes" else "Fake" for a in ans]
        return pred_label

    def PrintResult(self, detailed=False, acc=None, confusion_mat=None, ans_list=None, labels=None, logPath=None):
        
        if acc:
            self.acc = acc
        if confusion_mat:
            self.confusion_mat = confusion_mat
        if ans_list:
            self.ans_list = ans_list
        if labels:
            self.labels = labels
        
        if logPath:
            logfile = open(logPath, 'a')
        
        if detailed:
            
            print(f'[TIME]      : {time.ctime()}', file=logfile)
            print(f'[Finetuned] : {self.model.finetuned}', file=logfile)
            print(f'[Img roots] : {self.roots}', file=logfile)
            print(f'[Labels]    : {self.text_labels}', file=logfile)
            print(f'[Question]  : {self.question}\n', file=logfile)
            
            print(f'=== Overall ===', file=logfile)
            print(f'Acc: {self.acc*100:.2f}%', file=logfile)
            self.PrintConfusion(self.confusion_mat, logfile=logfile)
            print('\n', file=logfile)
            
            if 0 in self.labels:
                real_ans_list = self.ans_list[self.labels==0]
                real_label = [0] * len(real_ans_list)
                self.real_acc = accuracy_score(real_label, real_ans_list)
                self.real_confusion_mat = confusion_matrix(real_label, real_ans_list, labels=[0,1])
                print(f'=== Real images ===', file=logfile)
                print(f'Acc: {self.real_acc*100:.2f}%', file=logfile)
                self.PrintConfusion(self.real_confusion_mat, logfile=logfile)
                print('\n', file=logfile)
            else:
                print(f'=== No real images ===\n', file=logfile)
            
            
            if 1 in self.labels:
                fake_ans_list = self.ans_list[self.labels==1]
                fake_label = [1] * len(fake_ans_list)
                self.com_acc = accuracy_score(fake_label, fake_ans_list)
                self.com_confusion_mat = confusion_matrix(fake_label, fake_ans_list, labels=[0,1])
                print(f'=== Fake images ===', file=logfile)
                print(f'Acc: {self.com_acc*100:.2f}%', file=logfile)
                self.PrintConfusion(self.com_confusion_mat, logfile=logfile)
                print('\n', file=logfile)
            else:
                print(f'=== No fake images ===\n', file=logfile)
        else:
            print(f'Question: {self.question}\n', file=logfile)
            print(f'Acc: {self.acc*100:.2f}%', file=logfile)
            self.PrintConfusion(self.confusion_mat, logfile=logfile)
            print('\n', file=logfile)
        
        logfile.close()
    
    def PrintConfusion(self, mat, logfile):
        padding = ' '
        print(f'        | Pred real | Pred fake |', file=logfile)
        print(f'GT real | {mat[0, 0]:{padding}<{10}}| {mat[0, 1]:{padding}<{11}}|', file=logfile)
        print(f'GT fake | {mat[1, 0]:{padding}<{10}}| {mat[1, 1]:{padding}<{11}}|', file=logfile)
    
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis_root', type=str, default='/storage1/ruby/LAVIS/deepfake/annotations/test/ffhq-sladd.json', help='The path to the image directory.')
    parser.add_argument('--ann_paths', type=str, default='/storage1/ruby/LAVIS/deepfake/annotations/test/ffhq-sladd.json', help='The path to the annotation directory.')
    parser.add_argument('--log', type=str, default="log/log.txt", help='Path to the log file.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_name', type=str, default='blip2_vicuna_instruct')
    parser.add_argument('--model_type', type=str, default='vicuna7b')

    return parser.parse_args()

def main():
    
    args = arg_parser()
    
    #question = ' '.join(args.question)
    question = "Is this photo real? If not, why?"
    logPath = args.log
    device = args.device
    model_name = args.model_name
    model_type=args.model_type
    
    if dirname(logPath):
        makedirs(dirname(logPath), exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() and device=="cuda" else "cpu")
    model, vis_processors, txt_processors = load_model_and_preprocess(name=model_name, model_type=model_type, is_eval=True, device=device)
    
    print(f'Load model OK!')
    
    instruct = InstructBLIP()
    instruct.LoadModels(model, vis_processors, txt_processors, device)
    
    print(f'Log path: {logPath}')
    print(f'Question: {question}')
    
    instruct.LoadData(args.vis_root, args.ann_paths)
    instruct.QueryImgs_batch(question=question, true_string="yes", logPath=logPath)
        
        

if __name__ == '__main__':
    main()