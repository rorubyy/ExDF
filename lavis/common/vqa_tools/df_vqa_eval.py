"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

# coding=utf-8

__author__ = "aagrawal"

# This code is based on the code written by Tsung-Yi Lin for MSCOCO Python API available at the following link:
# (https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/eval.py).
import sys
import re
from lavis.common.vqa_tools.vqa_eval import VQAEval



class DF_VQAEval(VQAEval):
    def __init__(self, vqa=None, vqaRes=None, n=2):
        self.n = n
        self.accuracy = {}
        self.evalQA = {}
        self.evalQuesType = {}
        self.evalAnsType = {}
        self.vqa = vqa
        self.vqaRes = vqaRes
        if vqa is not None:
            self.params = {"question_id": vqa.getQuesIds()}

        
    def evaluate(self, quesIds=None):
        if quesIds == None:
            quesIds = [quesId for quesId in self.params["question_id"]]
        gts = {}
        res = {}
        for quesId in quesIds:
            gts[quesId] = self.vqa.qa[quesId]
            res[quesId] = self.vqaRes.qa[quesId]

        print("computing ACC")
        dataset_correct_counts = {}
        dataset_total_counts = {}

        for quesId in quesIds:
            resAns = res[quesId]["answer"].lower()
            gtAns = gts[quesId]["answers"][0]["answer"].lower()
            dataset = gts[quesId]["dataset"]
            
            if dataset not in dataset_correct_counts:
                dataset_correct_counts[dataset]=0
                dataset_total_counts[dataset]=0
            dataset_total_counts[dataset] += 1
            if ("real" in resAns and "real" in gtAns) or ("fake" in resAns and "fake" in gtAns):
                dataset_correct_counts[dataset] += 1

            
        for dataset in dataset_total_counts:
            # calculate ACC
            if dataset == "real":
                continue
            real_correct_count = dataset_correct_counts.get("real", 0)
            real_total_count = dataset_total_counts.get("real", 0)
            dataset_correct_count = dataset_correct_counts[dataset]
            dataset_total_count = dataset_total_counts[dataset]

            combined_correct_count = dataset_correct_count + real_correct_count
            combined_total_count = dataset_total_count + real_total_count

            accuracy = combined_correct_count / combined_total_count if combined_total_count > 0 else 0
            self.accuracy[dataset] = accuracy
            print(f"Accuracy for {dataset} : {accuracy:.4f}")

        overall_correct_count = sum(dataset_correct_counts.values())
        overall_total_count = sum(dataset_total_counts.values())
        overall_accuracy = overall_correct_count / overall_total_count if overall_total_count > 0 else 0

        print(f"Overall Accuracy: {overall_accuracy:.4f}")
        self.accuracy["overall"] = overall_accuracy

    def setAccuracy(self, F1):
        self.accuracy["overall"] = F1.mean().item()
        # self.accuracy["overall"] = round(100 * float(sum(accQA)) / len(accQA), self.n)
