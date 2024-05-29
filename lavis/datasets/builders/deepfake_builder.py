"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry
from lavis.datasets.datasets.deepfake_datasets import DeepfakeDataset, DeepfakeEvalDataset


@registry.register_builder("deepfake")
class DeepfakeBuilder(BaseDatasetBuilder):
    train_dataset_cls = DeepfakeDataset
    eval_dataset_cls = DeepfakeEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/deepfake/default.yaml",
    }
