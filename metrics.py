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

import collections

import torch
import torcheval.metrics


def binary_false_positive_rate(
    probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> float:
    """Calculate false positive rate of classifier.

    Args:
        probs (torch.Tensor): Predictive probabilities, must be 1D tensor of shape
          (n_examples,).
        targets (torch.Tensor): Target labels, must be (0, 1) for negative positive
          and 1D tensor of shape (n_examples,).
        threshold (float): Predict positive label when `probs` >= `threshold`.
    """
    predictions = torch.where(probs >= threshold, 1, 0)
    negatives = targets == 0
    n_false_positives = ((predictions == 1) & negatives).sum()
    return (n_false_positives / negatives.sum()).item()


def binary_clf_metrics(
    probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> dict[str, float]:
    """Calculate binary classification metrics

    Args:
        probs (torch.Tensor): Model probabilities for each example.
        targets (torch.Tensor): Ground truth targets for each example.
        threshold (float): Threshold to use when calculating prediction.

    Arguments `probs` and `targets` need to be on the same device.

    Returns:
        dict[str, float]: The calculated values.
    """
    # These functions work with logits as well, but easier to just enforce that
    # we use probabilities, it prevents confusions with `threshold`.
    precision = torcheval.metrics.functional.binary_precision(
        probs,
        targets,
        threshold=threshold,
    )
    recall = torcheval.metrics.functional.binary_recall(
        probs,
        targets,
        threshold=threshold,
    )
    f1 = torch.nan_to_num(2 * precision * recall / (precision + recall))
    precisions, recalls, thresholds = (
        torcheval.metrics.functional.binary_precision_recall_curve(
            probs,
            targets,
        )
    )
    return {
        "accuracy": torcheval.metrics.functional.binary_accuracy(
            probs,
            targets,
            threshold=threshold,
        ).item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "fpr": binary_false_positive_rate(
            probs.squeeze(),
            targets.squeeze(),
            threshold=threshold,
        ),
        # These methods give different results depending on whether you use
        # probabilities or logits due to how the threshold is computed within
        # the functions.
        "auprc": torcheval.metrics.functional.binary_auprc(probs, targets).item(),
    }


def binary_epoch_end_stats(
    train_logits: torch.Tensor,
    train_targets: torch.Tensor,
    val_logits: torch.Tensor,
    val_targets: torch.Tensor,
    test_logits: torch.Tensor,
    test_targets: torch.Tensor,
    threshold: float,
) -> dict[str, float]:

    train_probs = torch.nn.functional.sigmoid(train_logits)
    val_probs = torch.nn.functional.sigmoid(val_logits)
    test_probs = torch.nn.functional.sigmoid(test_logits)
    train_stats = binary_clf_metrics(
        train_probs.squeeze(),
        train_targets.squeeze(),
        threshold=threshold,
    )
    val_stats = binary_clf_metrics(
        val_probs.squeeze(),
        val_targets.squeeze(),
        threshold=threshold,
    )
    test_stats = binary_clf_metrics(
        test_probs.squeeze(),
        test_targets.squeeze(),
        threshold=threshold,
    )
    stats = {
        **{f"train/{k}": v for k, v in train_stats.items()},
        **{f"val/{k}": v for k, v in val_stats.items()},
        **{f"test/{k}": v for k, v in test_stats.items()},
    }
    return stats


def epoch_end_stats(
    train_unsafe_logits: torch.Tensor,
    train_category_logits: torch.Tensor,
    train_targets: torch.Tensor,
    val_unsafe_logits: torch.Tensor,
    val_category_logits: torch.Tensor,
    val_targets: torch.Tensor,
    test_unsafe_logits: torch.Tensor,
    test_category_logits: torch.Tensor,
    test_targets: torch.Tensor,
    threshold: float,
    category_names: list[str] | None = None,
) -> dict[str, float]:
    """Compute Lora-Guard training metrics.

    Args:
        train_unsafe_logits (torch.Tensor): Train unsafe logits for binary harmfulness label.
        train_category_logits (torch.Tensor): Train category logits for each harmfulness category.
        train_targets (torch.Tensor): Train targets.
        val_unsafe_logits (torch.Tensor): Validation unsafe logits for binary harmfulness label.
        val_category_logits (torch.Tensor): Validation category logits for each
            harmfulness category.
        val_targets (torch.Tensor): Validation targets.
        val_unsafe_logits (torch.Tensor): Test unsafe logits for binary harmfulness label.
        test_category_logits (torch.Tensor): Test category logits for each harmfulness category.
        test_targets (torch.Tensor): Test targets.
        threshold (float): Classification threshold.
        category_names (list[str] | None): Class names to append to statistic
            names.  Must have length equal to number of columns in the tensor
            inputs. Default to None, in which case we use integer indices.

    Returns:
        dict[str, float]: The computed statistics for each class.
    """
    category_names = category_names or list(
        map(str, range(train_category_logits.shape[1]))
    )

    # Converted to safe/unsafe
    binarized_multilabel_stats = binary_epoch_end_stats(
        train_logits=train_unsafe_logits,
        train_targets=train_targets.any(dim=1).to(dtype=int),
        val_logits=val_unsafe_logits,
        val_targets=val_targets.any(dim=1).to(dtype=int),
        test_logits=test_unsafe_logits,
        test_targets=test_targets.any(dim=1).to(dtype=int),
        threshold=threshold,
    )
    all_stats = dict()
    for k, v in binarized_multilabel_stats.items():
        split, rest = k.split("/", maxsplit=1)
        all_stats[f"{split}/binarized_{rest}"] = v

    for i, name in enumerate(category_names):
        cls_train_lgts = train_category_logits[:, i]
        cls_train_tgts = train_targets[:, i]
        cls_val_lgts = val_category_logits[:, i]
        cls_val_tgts = val_targets[:, i]
        cls_test_lgts = test_category_logits[:, i]
        cls_test_tgts = test_targets[:, i]

        stats = binary_epoch_end_stats(
            train_logits=cls_train_lgts,
            train_targets=cls_train_tgts.to(torch.int),
            val_logits=cls_val_lgts,
            val_targets=cls_val_tgts.to(torch.int),
            test_logits=cls_test_lgts,
            test_targets=cls_test_tgts.to(torch.int),
            threshold=threshold,
        )
        all_stats.update({f"{k}_{name}": v for k, v in stats.items()})
    return all_stats
