################################################################################
# Copyright (c) 2024 Samsung Electronics Co., Ltd.
#
# Author(s):
# Hayder Elesedy (b.elesedy@partner.samsung.com; hayder.elesedy@gmail.com)
#
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License, (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-sa/4.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# For conditions of distribution and use, see the accompanying LICENSE.md file.
################################################################################

import peft
import torch


def _get_last_non_pad_token_idx_pad_right(
    input_ids: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    # We score based on the last token features and assume that padding is done
    # on the right.
    return input_ids.eq(pad_token_id).to(dtype=torch.int).argmax(-1) - 1


def add_lora_adapters(
    base_model,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> peft.peft_model.PeftModelForFeatureExtraction:
    """Add LoRA adapters to `base_model` using some standardised parameter settings.

    Note that this modifies `base model`! This is due to the behaviour of the HuggingFace
    peft module.

    Args:
        base_model: model to add the lora adapters to. Note that this argument is modified!
        lora_r (int): LoRA r parameter.
        lora_alpha (int): LoRA alpha parameters, typically twice r.
        lora_dropout (float): LoRA dropout parameter.

    Returns:
        (peft.peft_model.PeftModelForFeatureExtraction): The model with adapters attached.
    """
    peft_config = peft.LoraConfig(
        task_type=peft.TaskType.FEATURE_EXTRACTION,
        peft_type=peft.PeftType.LORA,
        r=lora_r,
        # If the bias parameter is not 'none', then even with adapters disabled
        # the model will not produce the original non-adapted output. Hence,
        # we want this value only.
        bias="none",
        # Rule of thumb is 2x, according to Seb Raschka
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        # Kaiming uniform for A, zeros for B, see HF docs.
        init_lora_weights=True,
    )
    peft_model = peft.PeftModelForFeatureExtraction(
        base_model,
        peft_config,
    )
    return peft_model


class GuardModel(torch.nn.Module):
    """LoraGuard"""
    def __init__(
        self,
        lm_features: torch.nn.Module,
        feature_dim: int,
        n_categories: int,
        clf_head_bias: bool,
        pad_token_id: int,
        unsafe_head: bool = False,
    ):
        out_dim = n_categories + int(unsafe_head)
        if out_dim < 2:
            raise ValueError(
                "Must be at least one category. "
                "Must be more than one category if `unsafe_head=False`."
                f"Got `n_categories={n_categories}` and `unsafe_head={unsafe_head}`."
            )
        super().__init__()
        self.lm_features = lm_features
        self.clf_head = self._make_clf_head(
            in_dim=feature_dim,
            out_dim=out_dim,
            bias=clf_head_bias,
        )
        self.pad_token_id = pad_token_id
        self.unsafe_head = unsafe_head

    def set_requires_grad(self, require_grad: bool) -> "GuardModel":
        """Set requires_grad for all parameters of the model.

        Args:
            require_grad (bool): The desired value of requires_grad for
                all parameters in the model.

        Returns:
           model (GuardModel): This instance.
        """
        for p in self.parameters():
            p.requires_grad_(require_grad)
        return self

    @staticmethod
    def _make_clf_head(in_dim: int, out_dim: int, bias: bool):
        clf_head = torch.nn.Linear(in_dim, out_dim, bias=bias)
        torch.nn.init.xavier_uniform_(clf_head.weight)
        if clf_head.bias is not None:
            clf_head.bias.data.fill_(0.0)
        return clf_head

    def forward(
        self,
        input_ids,
        attention_mask,
        **lm_features_kwds,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lm_output = self.lm_features(input_ids, attention_mask, **lm_features_kwds)
        last_hidden_state = lm_output.last_hidden_state
        idx = _get_last_non_pad_token_idx_pad_right(input_ids, self.pad_token_id)
        features = last_hidden_state[torch.arange(last_hidden_state.shape[0]), idx, :]
        logits = self.clf_head(features)
        return self.split_unsafe_and_category_logits(logits)

    def split_unsafe_and_category_logits(
        self,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split logits from this model into logits for safe/unsafe and category predictions.

        Args:
            logits (torch.Tensor): Logits from this model.

        Returns:
            (unsafe_logits, category_logits) tuple[torch.Tensor, torch.Tensor]: The logits for safe/unsafe
                prediction and logits for category prediction.
        """
        if self.unsafe_head:
            unsafe_logits, category_logits = torch.split(
                logits,
                [1, logits.shape[-1] - 1],
                dim=-1,
            )
        else:
            unsafe_logits = logits.max(dim=-1, keepdims=True).values
            category_logits = logits
        return unsafe_logits, category_logits

    def lora_and_clf_head_state_dict(self) -> dict[str, torch.Tensor]:
        """Isolate adaptor parameters from LoraGuard state dict.

        This depends on HuggingFace naming convention for the lora parameters, should they
        exist.

        Returns:
            dict[str, torch.Tensor]: State dict of only the adaptor parameters.
        """
        return {
            k: v
            for k, v in self.state_dict().items()
            if ".lora_" in k or k.startswith("clf_head")
        }
