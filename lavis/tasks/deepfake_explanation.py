"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import json
import os

import lavis.common.dist_utils as dist_utils
from lavis.common.registry import registry
from lavis.common.vqa_tools.vqa import VQA, CustomVQA
from lavis.common.vqa_tools.vqa_eval import VQAEval, Cunstom_VQAEval
from lavis.tasks.base_task import BaseTask


@registry.register_task("deepfake_explanation")
class DFTask(BaseTask):
    def __init__(
        self,
        num_beams,
        max_len,
        min_len,
        evaluate,
        num_ans_candidates,
        inference_method="rank",
        prompt="",
    ):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len

        self.evaluate = evaluate
        self.inference_method = inference_method
        self.num_ans_candidates = num_ans_candidates
        self.prompt = prompt

        self.answer_list = None

        self.ques_files = dict()
        self.anno_files = dict()

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.get("num_beams", 3)
        max_len = run_cfg.get("max_len", 10)
        min_len = run_cfg.get("min_len", 1)

        evaluate = run_cfg.get("evaluate", False)

        inference_method = run_cfg.get("inference_method", "rank")
        num_ans_candidates = run_cfg.get("num_ans_candidates", 128)
        prompt = run_cfg.get("prompt", "")

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            num_ans_candidates=num_ans_candidates,
            inference_method=inference_method,
            prompt=prompt,
        )

    def build_datasets(self, cfg):
        datasets = super().build_datasets(cfg)
        for dataset in datasets.values():
            for split in dataset:
                annotation_paths = cfg.datasets_cfg.deepfake.build_info.annotations[
                    split
                ]["storage"]
                self.anno_files[split] = annotation_paths
                self.ques_files[split] = annotation_paths

        return datasets

    def valid_step(self, model, samples):
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt=self.prompt,
        )
        pred_qa_pairs = []

        question_id = samples["question_id"]
        for answer, ques_id in zip(answers, question_id):
            ques_id = int(ques_id.item())
            pred_qa_pairs.append({"question_id": ques_id, "answer": answer})

        return pred_qa_pairs

    def after_evaluation(self, val_result, split_name, mode=None, **kwargs):
        result_file = self.save_result(
            val_result,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_vqa_result",
            remove_duplicate="question_id",
        )

        metrics = self._report_metrics(
            result_file=result_file, split=split_name, mode=mode
        )
        return metrics

    @dist_utils.main_process
    def _report_metrics(self, result_file, split, mode=None):
        """
        Use official VQA evaluation script to report metrics.
        """
        metrics = {}

        if split in self.ques_files and split in self.anno_files:
            if mode == "val":
                vqa = CustomVQA(annotation_file=self.ques_files[mode])
                vqa_result = vqa.loadRes_custom(
                    resFile=result_file, quesFile=self.ques_files[mode]
                )
                vqa_scorer = Cunstom_VQAEval(vqa, vqa_result, n=2)
                logging.info("Start VQA evaluation.")
                vqa_scorer.evaluate()
                overall_acc = vqa_scorer.accuracy["overall"]
                metrics["agg_metrics"] = overall_acc
                logging.info("Overall Accuracy is: %.02f\n" % overall_acc)
            else:
                for index, ques_file in enumerate(self.ques_files[mode]):
                    vqa = CustomVQA(annotation_file=ques_file)
                    vqa_result = vqa.loadRes_custom(
                        resFile=result_file, quesFile=ques_file
                    )
                    vqa_scorer = Cunstom_VQAEval(vqa, vqa_result, n=2)
                    logging.info(
                        "Start Testing VQA evaluation for file: %s" % ques_file
                    )
                    vqa_scorer.evaluate()
                    logging.info(
                        "Overall Accuracy for %s is: %.02f\n"
                        % (ques_file, vqa_scorer.accuracy["overall"])
                    )

                    if index == 0:
                        overall_acc = vqa_scorer.accuracy["overall"]
                        metrics["agg_metrics"] = overall_acc
            # vqa = VQA(self.anno_files[split], self.ques_files[split])
            # vqa_result = vqa.loadRes(
            #     resFile=result_file, quesFile=self.ques_files[split]
            # )
            # # create vqaEval object by taking vqa and vqaRes
            # # n is precision of accuracy (number of places after decimal), default is 2
            # vqa_scorer = VQAEval(vqa, vqa_result, n=2)
            # logging.info("Start VQA evaluation.")
            # vqa_scorer.evaluate()

            # # print accuracies
            # overall_acc = vqa_scorer.accuracy["overall"]
            # metrics["agg_metrics"] = overall_acc

            # logging.info("Overall Accuracy is: %.02f\n" % overall_acc)
            # logging.info("Per Answer Type Accuracy is the following:")

            # for ans_type in vqa_scorer.accuracy["perAnswerType"]:
            #     logging.info(
            #         "%s : %.02f"
            #         % (ans_type, vqa_scorer.accuracy["perAnswerType"][ans_type])
            #     )
            #     metrics[ans_type] = vqa_scorer.accuracy["perAnswerType"][ans_type]

            # with open(
            #     os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
            # ) as f:
            #     f.write(json.dumps(metrics) + "\n")
        # TODO change to deepfake explination
        else:
            if mode == "val":
                ques_file = "/storage1/ruby/thesis_dataset/ann/test/generation/fixed_text_input/ip2p-test.json"
            else:
                ques_file = "/storage1/ruby/thesis_dataset/ann/test/generation/fixed_text_input/ip2p-test.json"

            vqa = CustomVQA(annotation_file=ques_file)
            vqa_result = vqa.loadRes_custom(resFile=result_file, quesFile=ques_file)
            # vqa_scorer = VQAEval(vqa, vqa_result, n=2)
            vqa_scorer = Cunstom_VQAEval(vqa, vqa_result, n=2)
            logging.info("Start VQA evaluation.")
            vqa_scorer.evaluate()
            overall_acc = vqa_scorer.accuracy["overall"]
            metrics["agg_metrics"] = overall_acc
            logging.info("Overall Accuracy is: %.02f\n" % overall_acc)

        return metrics
