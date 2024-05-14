"""
Requires Transformer 4.28 and above, implementation may change according the Llama implementation
"""

import logging
import string
from packaging import version

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

import transformers

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.soft_embedding import SoftEmbedding

import torch.nn.functional as F


@registry.register_model("blip2_vicuna_instruct")
class Blip2VicunaInstruct(Blip2Base):
    """
    BLIP2 Vicuna model.
    Supported model types:
        - vicuna7b
        - vicuna13b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_vicuna_instruct", "vicuna7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "vicuna7b": "configs/models/blip2/blip2_instruct_vicuna7b.yaml",
        "vicuna13b": "configs/models/blip2/blip2_instruct_vicuna13b.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        llm_model="",
        prompt="",
        max_txt_len=128,
        max_output_txt_len=256,
        apply_lemmatizer=False,
        qformer_text_input=True,
    ):
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse(
            "4.28"
        ), "BLIP-2 Vicuna requires transformers>=4.28"
        from transformers import LlamaTokenizer
        from lavis.models.blip2_models.modeling_llama import LlamaForCausalLM

        self.tokenizer = self.init_tokenizer(truncation_side="left")

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )

        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None
        # NEW set Qformersoft_embedding
        self.Qformer.soft_embedding = SoftEmbedding(self.Qformer.get_input_embeddings())
        self.Qformer.set_input_embeddings(self.Qformer.soft_embedding)

        self.llm_tokenizer = LlamaTokenizer.from_pretrained(
            llm_model, use_fast=False, truncation_side="left"
        )
        self.llm_model = LlamaForCausalLM.from_pretrained(
            llm_model, torch_dtype=torch.float16
        )
        self.llm_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.llm_tokenizer.add_special_tokens({"bos_token": "</s>"})
        self.llm_tokenizer.add_special_tokens({"eos_token": "</s>"})
        self.llm_tokenizer.add_special_tokens({"unk_token": "</s>"})
        # self.llm_tokenizer.pad_token = self.llm_tokenizer.unk_token

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        # self.eos_token_id = self.llm_tokenizer(
        #     self.llm_tokenizer.eos_token, add_special_tokens=False
        # ).input_ids[0]

        # NEW set LLM soft_embedding
        self.llm_model.soft_embedding = SoftEmbedding(
            self.llm_model.get_input_embeddings()
        ).to(torch.float32)
        self.llm_model.set_input_embeddings(self.llm_model.soft_embedding)

        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False
            if "learned_embedding" in name:
                param.requires_grad = True

        # ----- new ----- classification
        Qformer_hidden_size = self.Qformer.config.hidden_size
        llm_hidden_size = self.llm_model.config.hidden_size
        self.llm_proj = nn.Linear(Qformer_hidden_size, llm_hidden_size)
        num_classes = 2
        self.cls_head = nn.Sequential(
            nn.Linear(Qformer_hidden_size, Qformer_hidden_size),
            nn.ReLU(),
            nn.Linear(Qformer_hidden_size, num_classes),
        )
        # ----------
        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt
        prompt_tokens = self.llm_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self._lemmatizer = None

        self.qformer_text_input = qformer_text_input


    def set_current_epoch(self, epoch):
        self.current_epoch = epoch

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens["input_ids"].append(
                torch.cat(
                    [
                        input_ids[i][:this_input_ones],
                        output_ids[i][1:],
                        input_ids[i][this_input_ones:],
                    ]
                )
            )
            llm_tokens["attention_mask"].append(
                torch.cat(
                    [
                        input_atts[i][:this_input_ones],
                        output_atts[i][1:],
                        input_atts[i][this_input_ones:],
                    ]
                )
            )
        llm_tokens["input_ids"] = torch.stack(llm_tokens["input_ids"])
        llm_tokens["attention_mask"] = torch.stack(llm_tokens["attention_mask"])
        return llm_tokens, input_part_targets_len

    def modify_text_input_with_classification(self, prompt, classification_results):
        # This function assumes that `samples['text_input']` is a list of text inputs
        # and `classification_results` is a list of classification results ('real' or 'fake')
        modified_texts = []
        for text, classification in zip(prompt, classification_results):
            # Add a hint based on the classification
            hint = "Note: The image is considered " + classification + "."
            modified_text = text + " " + hint
            modified_texts.append(modified_text)
        return modified_texts

    def compute_contrastive_loss(self, anchor, positive, negative, tau=0.07):
        def cosine_similarity(x, y):
            x_norm = x / (torch.norm(x, dim=1, keepdim=True) + 1e-6)
            y_norm = y / (torch.norm(y, dim=1, keepdim=True) + 1e-6)
            return torch.sum(x_norm * y_norm, dim=1)

        positive_similarity = cosine_similarity(anchor, positive) / tau
        negative_similarity = cosine_similarity(anchor, negative) / tau

        logits = torch.cat(
            [positive_similarity.unsqueeze(1), negative_similarity.unsqueeze(1)], dim=1
        )

        labels = torch.zeros(logits.size(0), dtype=torch.long).to(anchor.device)
        loss = F.cross_entropy(logits, labels)
        return loss

    def set_current_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, samples):
        image = samples["image"]
        with self.maybe_autocast():
            final_output, intermediate_outputs = self.visual_encoder(image)
            image_embeds = self.ln_vision(final_output)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        def weighted_average_features(layers, weights):
            assert len(layers) == len(
                weights
            ), "The number of layers and weights must match"
            weighted_sum = sum(w * layer for w, layer in zip(weights, layers))
            return weighted_sum
        
        selected_layers_indices = [2, 7, 12, 18, 24, 30, 36]  
        selected_layers = [intermediate_outputs[i] for i in selected_layers_indices]
        fused_features = torch.cat(selected_layers, dim=-1) 
        adaptation_layer = nn.Linear(fused_features.shape[-1], image_embeds.shape[-1])
        fused_features = adaptation_layer(fused_features)

        # front_layers = intermediate_outputs[:3]
        # back_layers = intermediate_outputs[-3:]

        # front_weights = [1 / 3] * 3
        # back_weights = [1 / 3] * 3
        # front_features = weighted_average_features(front_layers, front_weights)
        # back_features = weighted_average_features(back_layers, back_weights)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                samples["text_input"],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                image.device
            )
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            fuse_query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=fused_features,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            #  ----- new ----- constrative learning loss
            if samples["positive_outputs"] and samples["negative_outputs"]:
                pos_text_Qformer = self.tokenizer(
                    samples["positive_outputs"],
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)
                pos_Qformer_atts = torch.cat([query_atts, pos_text_Qformer.attention_mask],dim=1)

                neg_text_Qformer = self.tokenizer(
                    samples["negative_outputs"],
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)
                neg_Qformer_atts = torch.cat([query_atts, neg_text_Qformer.attention_mask],dim=1)

                pos_query_output = self.Qformer.bert(
                    pos_text_Qformer.input_ids,
                    attention_mask=pos_Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=fused_features,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

                neg_query_output = self.Qformer.bert(
                    neg_text_Qformer.input_ids,
                    attention_mask=neg_Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=fused_features,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        # -----new ----- text contrastive learning loss
        cls_original = query_output.last_hidden_state[:, 0, :]
        cls_positive = pos_query_output.last_hidden_state[:, 0, :]
        cls_negative = neg_query_output.last_hidden_state[:, 0, :]
        contrastive_loss = self.compute_contrastive_loss(cls_original, cls_positive, cls_negative)
        # ---------

        # ----- new ----- classification
        # classification_targets = samples["label"]
        # label_to_index = {"real": 0, "fake": 1}
        # target_indices = [label_to_index[label] for label in classification_targets]

        # target_tensor = torch.tensor(target_indices).to("cuda:0")

        # classification_prediction = self.cls_head(
        #     class_query_output.last_hidden_state[:, 0, :]
        # )
        # classification_loss = F.cross_entropy(classification_prediction, target_tensor)
        # ----------

        # ----- new ----- get classification result
        # _, predicted_indices = torch.max(classification_prediction, dim=1)
        # classification_results = ['real' if idx == 0 else 'fake' for idx in predicted_indices]
        # modified_text_input = self.modify_text_input_with_classification(samples["text_input"], classification_results)
        # samples['text_input'] = modified_text_input

        # ----------

        inputs_llm = self.llm_proj(
            query_output.last_hidden_state[:, : query_tokens.size(1), :]
        )
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = "left"
        text_input_tokens = self.llm_tokenizer(
            samples["text_input"],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        self.llm_tokenizer.truncation_side = "right"
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in samples["text_output"]],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(image.device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        # do not apply loss to the padding
        targets = llm_tokens["input_ids"].masked_fill(
            llm_tokens["input_ids"] == self.llm_tokenizer.pad_token_id, -100
        )

        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        # do not apply loss to the query tokens
        empty_targets = (
            torch.ones(atts_llm.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        # NEW
        # FIXME TI
        empty_targets_soft_embedding = (
            torch.ones(
                [atts_llm.size(0), self.llm_model.soft_embedding.n_tokens],
                dtype=torch.long,
            )
            .to(image.device)
            .fill_(-100)
        )
        targets = torch.cat(
            [empty_targets, empty_targets_soft_embedding, targets], dim=1
        )
        # targets = torch.cat([empty_targets, targets], dim=1)
        # targets = torch.cat([empty_targets, targets, text_output_tokens.input_ids[:, 1].unsqueeze(1)], dim =1)

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens["input_ids"])


        inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        # NEW
        # FIXME TI
        extract_attention_mask = torch.ones(
            llm_tokens["input_ids"].size(0),
            self.llm_model.soft_embedding.n_tokens,
            device=llm_tokens["input_ids"].device,
        )
        attention_mask = torch.cat(
            [atts_llm, extract_attention_mask, llm_tokens["attention_mask"]], dim=1
        )
        # attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)

        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        vlm_loss = outputs.loss

        total_epochs = 10
        weight_classification = (1 - (self.current_epoch / total_epochs)) * 0.5
        weight_contrastive = (self.current_epoch / total_epochs) * 0.5

        # ----- new ----- classification
        # loss = vlm_loss + weight_classification * classification_loss
        # ----- new ----- contrastive loss
        loss = vlm_loss + contrastive_loss * weight_contrastive
        # loss = vlm_loss + weight_classification* classification_loss + weight_contrastive *contrastive_loss
        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=True,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        self.llm_tokenizer.padding_side = "left"

        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        image = samples["image"]

        bs = image.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert (
                len(prompt) == bs
            ), "The number of prompts must be equal to the batch size."

        # For TextCaps
        if "ocr_tokens" in samples.keys() and "{}" in prompt[0]:
            prompt = [
                p.format(", ".join(samples["ocr_tokens"][i][:30]))
                for i, p in enumerate(prompt)
            ]

        query_tokens = self.query_tokens.expand(bs, -1, -1)

        if self.qformer_text_input:
            # remove ocr tokens in q_former (for eval textvqa)
            # qformer_prompt = prompt
            # qformer_prompt = ['Question: ' + qp.split(' Question: ')[1] for qp in qformer_prompt]

            text_Qformer = self.tokenizer(
                prompt,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                image.device
            )
            # FIXME TI
            # add soft embedding mask
            soft_prompt_mask = torch.ones(
                [
                    text_Qformer.attention_mask.size(0),
                    self.Qformer.soft_embedding.n_tokens,
                ],
                dtype=torch.long,
            ).to(image.device)
            extended_mask = torch.cat(
                [text_Qformer.attention_mask, soft_prompt_mask], dim=1
            )
            Qformer_atts = torch.cat([query_atts, extended_mask], dim=1)
            # Qformer_atts = torch.cat([query_atts, text_Qformer.attention_masktext_Qformer.attention_mask], dim=1)

        # For video data
        if image.dim() == 5:
            inputs_llm, atts_llm = [], []
            for j in range(image.size(2)):
                this_frame = image[:, :, j, :, :]
                with self.maybe_autocast():
                    final_output, intermediate_outputs = self.visual_encoder(this_frame)
                    frame_embeds = self.ln_vision(final_output)
                frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(
                    image.device
                )

                if self.qformer_text_input:
                    frame_query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                else:
                    frame_query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                frame_inputs_llm = self.llm_proj(
                    frame_query_output.last_hidden_state[:, : query_tokens.size(1), :]
                )
                frame_atts_llm = torch.ones(
                    frame_inputs_llm.size()[:-1], dtype=torch.long
                ).to(image.device)
                inputs_llm.append(frame_inputs_llm)
                atts_llm.append(frame_atts_llm)
            inputs_llm = torch.cat(inputs_llm, dim=1)
            atts_llm = torch.cat(atts_llm, dim=1)
        else:
            with self.maybe_autocast():
                final_output, intermediate_outputs = self.visual_encoder(image)
                image_embeds = self.ln_vision(final_output)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            if self.qformer_text_input:
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            # ----- new ----- add classification result to llm input prmopt
            # classification_prediction = self.cls_head(query_output.last_hidden_state[:, 0, :])
            # _, predicted_indices = torch.max(classification_prediction, dim=1)
            # classification_results = ['real' if idx == 0 else 'fake' for idx in predicted_indices]
            # modified_text_input = self.modify_text_input_with_classification(prompt, classification_results)
            # prompt = modified_text_input
            # ----------

            inputs_llm = self.llm_proj(
                query_output.last_hidden_state[:, : query_tokens.size(1), :]
            )
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(
                image.device
            )

        llm_tokens = self.llm_tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).to(image.device)

        with self.maybe_autocast():

            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            # NEW
            soft_prompt_mask = torch.ones(
                [inputs_embeds.size(0), self.llm_model.soft_embedding.n_tokens],
                dtype=torch.long,
            ).to(image.device)
            attention_mask = torch.cat(
                [atts_llm, soft_prompt_mask, llm_tokens.attention_mask], dim=1
            )
            attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                # eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

        outputs[outputs == 0] = 2  # convert output id 0 to 2 (eos_token_id)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return output_text

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=0,
        **kwargs,
    ):
        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        if prompt:
            if prompt.count("{}") == 2:
                if "ocr_tokens" in samples:
                    text_input = [
                        prompt.format(
                            ", ".join(samples["ocr_tokens"][i][:30]),
                            samples["text_input"][i],
                        )
                        for i in range(len(samples["text_input"]))
                    ]
                elif "choices" in samples:
                    text_input = []
                    for i in range(len(samples["text_input"])):
                        this_choices = [
                            f"({string.ascii_lowercase[j]}) {ch}"
                            for j, ch in enumerate(samples["choices"][i])
                        ]
                        this_choices = " ".join(this_choices)
                        text_input.append(
                            prompt.format(samples["text_input"][i], this_choices)
                        )
            else:
                text_input = [
                    prompt.format(question) for question in samples["text_input"]
                ]
        else:
            text_input = samples["text_input"]

        samples["prompt"] = text_input

        # output_text = self.generate(
        #     samples,
        #     num_beams=num_beams,
        #     max_length=max_len,
        #     min_length=min_len,
        #     length_penalty=length_penalty
        # )

        output_text = self.generate(
            samples,
            num_beams=num_beams,
            max_length=50,
            min_length=min_len,
            length_penalty=length_penalty,
        )

        if "apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]:
            output_text = self._lemmatize(output_text)

        return output_text

    def predict_class(
        self,
        samples,
        candidates,
        n_segments=1,
    ):
        self.llm_tokenizer.padding_side = "left"

        # If candidates is a list of lists, each sample has its candidates, then we need to iterate one by one
        if type(candidates[0]) == list:
            results = []

            for i in range(samples["image"].size(0)):
                this_sample = {
                    "image": samples["image"][i].unsqueeze(0),
                    "prompt": samples["prompt"],
                }

                if "text_input" in samples.keys():
                    this_sample["text_input"] = [samples["text_input"][i]]

                if "context" in samples.keys():
                    this_sample["context"] = [samples["context"][i]]

                if "history" in samples.keys():
                    this_sample["history"] = [samples["history"][i]]

                if "caption" in samples.keys():
                    this_sample["caption"] = [samples["caption"][i]]

                this_result = self._predict_class(
                    this_sample, candidates[i], n_segments
                )
                results.append(this_result)

            try:
                results = torch.cat(results, dim=0)
            except:
                results = [res.tolist()[0] for res in results]

            return results

        return self._predict_class(samples, candidates, n_segments)

    def _predict_class(
        self,
        samples,
        candidates,
        n_segments=1,
    ):
        image = samples["image"]
        prompt = samples["prompt"]

        bs = image.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert (
                len(prompt) == bs
            ), "The number of prompts must be equal to the batch size."

        if "text_input" in samples.keys():
            if type(samples["text_input"][0]) == list:
                prompt = [
                    prompt[i].format(*samples["text_input"][i])
                    for i in range(len(prompt))
                ]
            else:
                prompt = [
                    prompt[i].format(samples["text_input"][i])
                    for i in range(len(prompt))
                ]

        # scienceqa
        if "context" in samples.keys() and samples["context"] != "":
            prompt = [
                f'context: {samples["context"][i]}. {prompt[i]}'
                for i in range(len(prompt))
            ]

        # visual dialog
        if "history" in samples.keys() and samples["history"][0] != "":
            prompt = [
                f'dialog history: {samples["history"][i]}\n{prompt[i]}'
                for i in range(len(prompt))
            ]

        if "caption" in samples.keys() and samples["caption"][0] != "":
            prompt = [
                f'This image has the caption "{samples["caption"][i]}". {prompt[i]}'
                for i in range(len(prompt))
            ]

        query_tokens = self.query_tokens.expand(bs, -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                prompt,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                image.device
            )
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        if image.dim() == 5:
            inputs_llm, atts_llm = [], []
            for j in range(image.size(2)):
                this_frame = image[:, :, j, :, :]
                with self.maybe_autocast():
                    final_output, intermediate_outputs = self.visual_encoder(this_frame)
                    frame_embeds = self.ln_vision(final_output)
                    frame_atts = torch.ones(
                        frame_embeds.size()[:-1], dtype=torch.long
                    ).to(image.device)

                if self.qformer_text_input:
                    frame_query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                else:
                    frame_query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )

                frame_inputs_llm = self.llm_proj(
                    frame_query_output.last_hidden_state[:, : query_tokens.size(1), :]
                )
                frame_atts_llm = torch.ones(
                    frame_inputs_llm.size()[:-1], dtype=torch.long
                ).to(image.device)
                inputs_llm.append(frame_inputs_llm)
                atts_llm.append(frame_atts_llm)
            inputs_llm = torch.cat(inputs_llm, dim=1)
            atts_llm = torch.cat(atts_llm, dim=1)
        else:
            with self.maybe_autocast():
                image_embeds, intermediate_outputs = self.ln_vision(
                    self.visual_encoder(image)
                )
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            if self.qformer_text_input:
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            inputs_llm = self.llm_proj(
                query_output.last_hidden_state[:, : query_tokens.size(1), :]
            )
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(
                image.device
            )

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = "left"
        text_input_tokens = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            # truncation=True,
            # max_length=self.max_txt_len,
        ).to(image.device)

        empty_targets = (
            torch.ones(atts_llm.size(), dtype=torch.long).to(image.device).fill_(-100)
        )

        # self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = "right"
        n_cands = len(candidates)
        with self.maybe_autocast(dtype=torch.bfloat16):
            all_losses = []
            for n in range(n_segments):
                seg_len = n_cands // n_segments
                if n == (n_segments - 1):
                    seg_len = n_cands - seg_len * (n_segments - 1)

                start_i = n * (n_cands // n_segments)
                end_i = start_i + seg_len

                this_output_tokens = self.llm_tokenizer(
                    candidates[start_i:end_i],
                    return_tensors="pt",
                    padding="longest",
                    # truncation=True,
                    # max_length=self.max_output_txt_len,
                ).to(image.device)

                this_input_tokens_ids = text_input_tokens.input_ids.repeat_interleave(
                    seg_len, dim=0
                )
                this_input_tokens_atts = (
                    text_input_tokens.attention_mask.repeat_interleave(seg_len, dim=0)
                )

                this_output_tokens_ids = this_output_tokens.input_ids.repeat(bs, 1)
                this_output_tokens_atts = this_output_tokens.attention_mask.repeat(
                    bs, 1
                )

                this_llm_tokens, this_input_targets_len = self.concat_text_input_output(
                    this_input_tokens_ids,
                    this_input_tokens_atts,
                    this_output_tokens_ids,
                    this_output_tokens_atts,
                )

                this_llm_input_ids = this_llm_tokens["input_ids"]
                this_llm_atts = this_llm_tokens["attention_mask"]
                # this_llm_input_ids = torch.cat([this_input_tokens_ids, this_output_tokens_ids], dim=1)
                # this_llm_atts = torch.cat([this_input_tokens_atts, this_output_tokens_atts], dim=1)

                inputs_embeds = self.llm_model.get_input_embeddings()(
                    this_llm_input_ids
                )

                inputs_embeds = torch.cat(
                    [inputs_llm.repeat_interleave(seg_len, dim=0), inputs_embeds], dim=1
                )
                attention_mask = torch.cat(
                    [atts_llm.repeat_interleave(seg_len, dim=0), this_llm_atts], dim=1
                )

                this_targets = this_llm_input_ids.masked_fill(
                    this_llm_input_ids == self.llm_tokenizer.pad_token_id, -100
                )
                # this_targets[:, :this_input_tokens_ids.size(1)] = -100
                for i, l in enumerate(this_input_targets_len):
                    this_targets[i][:l] = -100

                this_targets = torch.cat(
                    [empty_targets.repeat_interleave(seg_len, dim=0), this_targets],
                    dim=1,
                )

                outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=this_targets,
                    reduction="none",
                )

                loss = outputs.loss

                loss = loss.reshape(bs, seg_len)
                # output_class_ranks = torch.argsort(loss, dim=-1)
                all_losses.append(loss)

            all_losses = torch.cat(all_losses, dim=-1)
            output_class_ranks = torch.argsort(all_losses, dim=-1)

        return output_class_ranks

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llm_model = cfg.get("llm_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 256)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        qformer_text_input = cfg.get("qformer_text_input", True)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            llm_model=llm_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            qformer_text_input=qformer_text_input,
        )

        # if qformer_text_input:
        #     # Hard-coded to load from BLIP-2 stage-1 pre-trained model (not ideal)
        #     model.load_from_pretrained(
        #         url_or_filename="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth"
        #     )

        model.load_checkpoint_from_config(cfg)

        return model
