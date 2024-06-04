"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

__author__ = "aagrawal"
__version__ = "0.9"

# Interface for accessing the VQA dataset.

# This code is based on the code written by Tsung-Yi Lin for MSCOCO Python API available at the following link:
# (https://github.com/pdollar/coco/blob/master/PythonAPI/pycocotools/coco.py).

# The following functions are defined:
#  VQA        - VQA class that loads VQA annotation file and prepares data structures.
#  getQuesIds - Get question ids that satisfy given filter conditions.
#  getImgIds  - Get image ids that satisfy given filter conditions.
#  loadQA     - Load questions and answers with the specified question ids.
#  showQA     - Display the specified questions and answers.
#  loadRes    - Load result file and create result object.

# Help on each function can be accessed by: "help(COCO.function)"

import json
import datetime
import copy
from lavis.common.vqa_tools.vqa import VQA



class DFVQA(VQA):
    def __init__(self, annotation_file=None):
        """
        Constructor of VQA helper class for reading and visualizing questions and answers.
        :param annotation_file (str): location of VQA annotation file
        :return:
        """
        # load dataset
        self.dataset = {}
        self.questions = {}
        self.qa = {}
        self.qqa = {}
        self.imgToQA = {}
        if not annotation_file == None:
            print("loading VQA annotations and questions into memory...")
            data = json.load(open(annotation_file, "r"))
            self.questions["questions"] = [{"question": item["text_input"], "question_id": item["question_id"]} for item in data]
            self.dataset["annotations"] = [
                {
                    "question_id": item["question_id"],
                    "attribute": set(item["attribute"]), 
                    "answers": [{"answer": item["text_output"]}],
                    "image_id": item["image"].split('/')[-1],
                    "dataset": item["dataset"]
                }
                for item in data
            ]            
            self.createIndex()
            

    def loadRes(self, resFile, quesFile):
        """
        Load result file and return a result object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = DFVQA()
        val_data = json.load(open(quesFile))
        res.questions["questions"] = [{"question": item["text_input"], "question_id": item["question_id"]} for item in val_data]
        
        time_t = datetime.datetime.utcnow()
        print("Loading and preparing results...     ")

        anns = json.load(open(resFile))
        assert type(anns) == list, "results is not an array of objects"
                
        for ann in anns:
            quesId = ann["question_id"]
            corresponding_val = next((item for item in val_data if item["question_id"] == quesId), None)

            if corresponding_val is not None:
                ann["image_id"] = corresponding_val["image"].split('/')[-1]
                ann["question_type"] = "vqa" 
                ann["answer_type"] = "vqa"  
            ann["answers"] = [{"answer": ann["answer"]}]

        print(
            "DONE (t=%0.2fs)" % ((datetime.datetime.utcnow() - time_t).total_seconds())
        )
        res.dataset["annotations"] = anns
        res.createIndex()
        return res