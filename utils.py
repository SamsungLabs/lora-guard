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

import dataclasses
import json
import typing

import torch
import tqdm

CallbackReturnType = dict[str, float] | None


def _calculate_binary_pos_weight(labels: torch.Tensor) -> torch.Tensor:
    """Calculate `pos_weight` for use in PyTorch binary cross-entropy loss.

    See, e.g., https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss.

    Args:
        labels (torch.Tensor): Tensor of labels, 1 for positive examples
          and 0 for negative examples.

    Returns:
        torch.Tensor: The weights for each class, pos_weight = n_neg / n_pos.
    """
    return 1 / labels.float().mean(axis=0) - 1  # type:ignore


def _unsafe_label_from_category_labels(labels: torch.Tensor) -> torch.Tensor:
    return labels.any(dim=-1, keepdims=True)


def make_loss_fn_binary_cross_entropy(train_labels: torch.Tensor):
    """Make loss function.

    Args:
        train_labels (torch.Tensor): The labels for the training set.
          Must have dim (n_examples, n_categories).
    """
    unsafe_pos_weight = _calculate_binary_pos_weight(
        _unsafe_label_from_category_labels(train_labels)
    )
    category_pos_weight = _calculate_binary_pos_weight(train_labels)

    def loss_fn(
        outputs: tuple[torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Loss function for LoraGuard training.

        Args:
            outputs (tuple[torch.Tensor, torch.Tensor]): Tuple of output logits,
                first element for unsafe prediction and second for category
                prediction.
            targets (torch.Tensor): The targets.

        Returns:
            (torch.Tensor): The loss values per datum.
        """
        unsafe_pred, category_preds = outputs
        unsafe_label = _unsafe_label_from_category_labels(targets)
        safety_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            unsafe_pred,
            unsafe_label.float(),
            reduction="none",
            pos_weight=unsafe_pos_weight.to(targets.device),
        )
        category_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            category_preds,
            targets.float(),
            reduction="none",
            pos_weight=category_pos_weight.to(targets.device),
        )
        per_category_weight = 0.5 / category_loss.shape[1]
        return torch.column_stack(
            [0.5 * safety_loss, per_category_weight * category_loss],
        )

    return loss_fn


@dataclasses.dataclass
class TrainingState:
    model: torch.nn.Module
    loss: float
    epoch: int
    global_step: int
    local_step: int

    def update(self, **updates) -> "TrainingState":
        return dataclasses.replace(self, **updates)


def train(
    model: torch.nn.Module,
    step_fn: typing.Callable[[typing.Any, TrainingState], TrainingState],
    dataloader: torch.utils.data.DataLoader,
    n_epochs: int,
    on_step_end: typing.Callable[[TrainingState], CallbackReturnType] | None = None,
    on_epoch_end: typing.Callable[[TrainingState], CallbackReturnType] | None = None,
    notebook: bool = False,
    disable_pbar: bool = False,
) -> tuple[list[tuple[int, CallbackReturnType]], list[tuple[int, CallbackReturnType]]]:

    # state will go into step fn with model, epoch and step correct and loss from
    # previous iteration expect nan on first iter.
    # Step fn will need to update the loss to be whatever it was after that step

    step_results = []
    epoch_results = []

    state = TrainingState(
        model, loss=float("nan"), epoch=-1, global_step=-1, local_step=-1
    )
    for epoch in range(n_epochs):
        dl_pbar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch}", disable=disable_pbar)
        for local_step, example in enumerate(dl_pbar):
            state = state.update(
                epoch=epoch,
                global_step=state.global_step + 1,
                local_step=local_step,
            )
            state = step_fn(example, state)
            if on_step_end is not None:
                step_results.append((state.global_step, on_step_end(state)))
        if on_epoch_end is not None:
            epoch_results.append((state.epoch, on_epoch_end(state)))
    return step_results, epoch_results


def maybe_add_pad_token_eos_token(tokenizer) -> None:
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("tokenizer.eos_token is None.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id


def parameter_counts(model: torch.nn.Module) -> tuple[int, int]:
    count = 0
    trainable_count = 0
    for p in model.parameters():
        count += p.nelement()
        if p.requires_grad:
            trainable_count += p.nelement()
    return count, trainable_count


def save_json(obj, fpath):
    with open(fpath, "w") as f:
        json.dump(obj, f)

def load_json(fpath) -> dict[str, typing.Any]:
    with open(fpath, "r") as f:
        return json.load(f)

