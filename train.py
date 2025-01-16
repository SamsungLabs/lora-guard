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

"""Distributed training of Lora-Guard on Beavertails30k"""

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import random

import accelerate
import datasets
import numpy as np
import pandas as pd
import torch
import tqdm
import transformers

import data
import lora_guard
import metrics
import utils


MODEL_DTYPE = torch.bfloat16
CLF_THRESHOLD = 0.5

# We have tested that our implementation is compatible with the
# following huggingface models.
# In principle, LoRA-Guard can be applied to any transformer based LLM, but
# implementations of the LLMs in HuggingFace may differ.
MODEL_CHOICES = [
    "meta-llama/Llama-2-7b-chat-hf",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]


def cmd_args() -> argparse.Namespace:
    class CheckNumericGreaterThanZero(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if values <= 0:
                raise argparse.ArgumentError(
                    self,
                    f"Argument value should be greater than 0, got {values}.",
                )
            setattr(namespace, self.dest, values)

    parser = argparse.ArgumentParser("Train LoRA-Guard on Beavertails30k")
    parser.add_argument(
        "hf_model_id",
        metavar="hf_model_id",
        type=str,
        choices=MODEL_CHOICES,
        help="HuggingFace ID of the chat model to use."
        " In principle any transformer based LLM is possible, but we"
        " have only verified our implementation against the models shown here.",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Where to save the metrics, model checkpoints and training outputs.",
    )
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=2,
        help="Train batch size (per device).",
        action=CheckNumericGreaterThanZero,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate for AdamW.",
        action=CheckNumericGreaterThanZero,
    )
    parser.add_argument(
        "--no-unsafe-head",
        action="store_true",
        help="Remove the additional trainable output head for unsafe/safe "
        "classification. Doing this will mean that overall safety labels in "
        " problems with multiple harm categories will be predicted by"
        " whether any of the harm categories are violated. This corresponds"
        " to v1 of the LoRA-Guard paper.",
    )
    parser.add_argument(
        "--no-clf-head-bias",
        action="store_true",
        help="Deactivate bias for linear output head of guard model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="How many epochs to train for.",
        action=CheckNumericGreaterThanZero,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for RNG.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of steps over which to accumulate gradients.",
        action=CheckNumericGreaterThanZero,
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=50,
        help="Evaluation batch size.",
        action=CheckNumericGreaterThanZero,
    )

    lora_args = parser.add_argument_group("LoRA parameters")
    lora_args.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank parameter.",
        action=CheckNumericGreaterThanZero,
    )
    lora_args.add_argument(
        "--lora-alpha",
        type=float,
        default=16,
        help="LoRA alpha parameter. A good rule of thumb is 2x lora-r.",
    )
    lora_args.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout parameter.",
        action=CheckNumericGreaterThanZero,
    )
    args = parser.parse_args()
    return args


@torch.inference_mode()
def get_logits_targets(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    accelerator: accelerate.Accelerator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate logits and targets across dataset in distributed setting.

    Args:
        model (torch.nn.Module): Model to do forward pass, assume to be in
          eval mode and prepared by `accelerator`.
        dataloader (torch.utils.data.DataLoader): DataLoader to sweep, should
          be prepared by `accelerator`.
        accelerator (accelerate.Accelerator): Accelerator instance to handle
          distributed computation.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Unsafe logits, category logits and targets.
    """
    all_targets = []
    all_unsafe_logits = []
    all_category_logits = []
    pbar = tqdm.tqdm(
        dataloader,
        disable=not accelerator.is_main_process,
        desc="Evaluation",
    )
    for example in pbar:
        targets = example["targets"].to(accelerator.device)
        input_ids = example["input_ids"].to(accelerator.device)
        attention_mask = example["attention_mask"].to(accelerator.device)
        unsafe_logits, category_logits = model(input_ids, attention_mask)
        targets, unsafe_logits, category_logits = accelerator.gather_for_metrics(
            [targets, unsafe_logits, category_logits]
        )
        all_targets.append(targets)
        all_unsafe_logits.append(unsafe_logits)
        all_category_logits.append(category_logits)
    unsafe_logits = torch.row_stack(all_unsafe_logits)
    category_logits = torch.row_stack(all_category_logits)
    targets = torch.row_stack(all_targets)
    return unsafe_logits, category_logits, targets


if __name__ == "__main__":
    datasets.disable_progress_bar()
    args = cmd_args()
    os.makedirs(args.output_dir, exist_ok=True)

    accelerate.utils.set_seed(args.seed, deterministic=False)
    torch.use_deterministic_algorithms(True, warn_only=False)

    def seed_dataloader_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    rng = torch.Generator("cpu")
    rng.manual_seed(args.seed)

    ga_plugin = accelerate.utils.GradientAccumulationPlugin(
        sync_with_dataloader=False,
        num_steps=args.gradient_accumulation_steps,
        adjust_scheduler=True,
        sync_each_batch=False,
    )

    dataloader_config = accelerate.DataLoaderConfiguration(
        split_batches=False,
        dispatch_batches=False,
        even_batches=True,
        use_seedable_sampler=True,
    )

    accelerator = accelerate.Accelerator(
        device_placement=False,
        gradient_accumulation_plugin=ga_plugin,
        rng_types=["torch", "generator", "cuda"],
    )

    def print_from_main_process(*args):
        if accelerator.is_main_process:
            return print(*args)

    # This is intentional, I want the script to error if we accidentally log to neptune
    # from the wrong process. An alternative would be to give the non-main processes a run
    # with debug=True.
    if accelerator.is_main_process:
        parameters = dict(vars(args))
        extra_args = {
            "learning_rate": args.learning_rate,
            "effective_batch_size": args.per_device_batch_size
            * args.gradient_accumulation_steps
            * accelerator.num_processes,
            "num_processes": accelerator.num_processes,
        }
        parameters.update(extra_args)
        utils.save_json(parameters, fpath=os.path.join(args.output_dir, "config.json"))

        checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

    print_from_main_process("Loading model and tokenizer.")
    chat_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.hf_model_id,
        # Disabling attention cache should save memory, and we don't need
        # it since we only do one forward pass at a time (not autoregressive
        # generation).
        use_cache=False,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.hf_model_id,
        padding_side="right",
    )
    utils.maybe_add_pad_token_eos_token(tokenizer)

    def get_labels(train_ds, key: str = "targets"):
        """Returns labels for the entire dataset."""
        return torch.tensor(train_ds[:][key], requires_grad=False)

    with accelerator.main_process_first():
        print_from_main_process("Loading and preprocessing data.")
        dataset = data.BeaverTails(tokenizer=tokenizer)
        train_ds, val_ds, test_ds = dataset.train_val_test_splits(rng)
        loss_fn = utils.make_loss_fn_binary_cross_entropy(get_labels(train_ds))

    if accelerator.is_main_process:
        torch.save(train_ds.indices, os.path.join(args.output_dir, "train_idx.pt"))
        torch.save(val_ds.indices, os.path.join(args.output_dir, "val_idx.pt"))

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        generator=rng,
        collate_fn=transformers.default_data_collator,
        worker_init_fn=seed_dataloader_worker,
    )
    eval_train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=transformers.default_data_collator,
    )
    eval_val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=transformers.default_data_collator,
    )
    eval_test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=transformers.default_data_collator,
    )

    print_from_main_process("Making guard model and moving to device.")

    peft_model = lora_guard.add_lora_adapters(
        chat_model.model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    guard_model = (
        lora_guard.GuardModel(
            lm_features=peft_model,
            pad_token_id=tokenizer.pad_token_id,
            feature_dim=chat_model.lm_head.in_features,
            n_categories=dataset.n_categories,
            clf_head_bias=not args.no_clf_head_bias,
            unsafe_head=not args.no_unsafe_head,
        )
        .to(dtype=MODEL_DTYPE, device=accelerator.device)
        .train()
    )

    if accelerator.is_main_process:
        param_count, trainable_parameters = utils.parameter_counts(guard_model)
        # Logging these values as strings because neptune does not support
        # int64 logging, so total param count will overflow.
        parameter_counts = {
            "n_total_parameters": str(param_count),
            "n_trainable_parameters": str(trainable_parameters),
        }
        utils.save_json(
            parameter_counts,
            os.path.join(args.output_dir, "parameter_counts.json"),
        )

    @torch.inference_mode()
    def on_step_end(state: utils.TrainingState) -> dict[str, float] | None:
        if accelerator.is_main_process:
            # We only log the loss from the main process, but it is averaged
            # across all examples on all processes in the train step.
            return {"train/loss": state.loss}

    @torch.inference_mode()
    def on_epoch_end(
        state: utils.TrainingState,
    ) -> utils.CallbackReturnType:

        accelerator.wait_for_everyone()
        state.model.eval()
        train_unsafe_logits, train_category_logits, train_targets = get_logits_targets(
            state.model,
            eval_train_loader,
            accelerator,
        )
        val_unsafe_logits, val_category_logits, val_targets = get_logits_targets(
            state.model,
            eval_val_loader,
            accelerator,
        )
        test_unsafe_logits, test_category_logits, test_targets = get_logits_targets(
            state.model,
            eval_test_loader,
            accelerator,
        )
        state.model.train()

        split_logits_targets = [
            ("train", train_unsafe_logits, train_category_logits, train_targets),
            ("val", val_unsafe_logits, val_category_logits, val_targets),
            ("test", test_unsafe_logits, test_category_logits, test_targets),
        ]
        if accelerator.is_main_process:
            for split, unsafe_logits, category_logits, targets in split_logits_targets:
                logits_and_targets_dir = os.path.join(
                    args.output_dir, "logits_and_targets"
                )
                os.makedirs(logits_and_targets_dir, exist_ok=True)

                torch.save(
                    unsafe_logits,
                    os.path.join(
                        logits_and_targets_dir,
                        f"{split}_unsafe_logits_epoch_{state.epoch}.pt",
                    ),
                )
                torch.save(
                    category_logits,
                    os.path.join(
                        logits_and_targets_dir,
                        f"{split}_category_logits_epoch_{state.epoch}.pt",
                    ),
                )
                torch.save(
                    targets,
                    os.path.join(
                        logits_and_targets_dir,
                        (
                            f"{split}_targets_epoch_{state.epoch}.pt"
                            if split == "train"
                            else f"{split}_targets.pt"
                        ),
                    ),
                )

            stats = metrics.epoch_end_stats(
                train_unsafe_logits,
                train_category_logits,
                train_targets,
                val_unsafe_logits,
                val_category_logits,
                val_targets,
                test_unsafe_logits,
                test_category_logits,
                test_targets,
                threshold=CLF_THRESHOLD,
                category_names=dataset.category_names,
            )

            adaptor_state_dict = accelerator.unwrap_model(
                state.model
            ).lora_and_clf_head_state_dict()
            torch.save(
                adaptor_state_dict,
                os.path.join(
                    checkpoint_dir, f"adaptor_state_dict_epoch_{state.epoch}.pt"
                ),
            )
            return stats

    optimizer = torch.optim.AdamW(guard_model.parameters(), lr=args.learning_rate)

    (
        guard_model,
        optimizer,
        train_loader,
        eval_train_loader,
        eval_val_loader,
        eval_test_loader,
    ) = accelerator.prepare(
        guard_model,
        optimizer,
        train_loader,
        eval_train_loader,
        eval_val_loader,
        eval_test_loader,
    )

    def train_step(
        example,
        state: utils.TrainingState,
    ) -> utils.TrainingState:
        input_ids = example["input_ids"].to(accelerator.device)
        attention_mask = example["attention_mask"].to(accelerator.device)
        targets = example["targets"].to(accelerator.device)
        outputs = state.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        with accelerator.accumulate(state.model):
            full_loss = loss_fn(outputs, targets.float())
            total_batch_size = (
                accelerator.num_processes
                * args.gradient_accumulation_steps
                * input_ids.shape[0]
            )
            loss = full_loss.sum() / total_batch_size
            full_loss = full_loss.detach()
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
        # Send loss averages across all examples across all GPUs.
        # The wait is unfortunate, but I think necessary to be exact.
        accelerator.wait_for_everyone()
        global_loss = accelerator.gather_for_metrics(full_loss).mean().item()
        return state.update(loss=global_loss)

    step_stats, epoch_stats = utils.train(
        model=guard_model,
        step_fn=train_step,
        dataloader=train_loader,
        n_epochs=args.epochs,
        on_step_end=on_step_end,
        on_epoch_end=on_epoch_end,
        disable_pbar=not accelerator.is_main_process,
    )

    def make_df(stats):
        counter, data = map(list, zip(*stats))
        return pd.DataFrame(data, index=counter)

    if accelerator.is_main_process:
        # The headline metrics are binarized_
        metrics_df = make_df(epoch_stats)
        metrics_df.index.name = "epoch"
        metrics_df.to_csv(os.path.join(args.output_dir, "metrics.csv"))

        train_losses = make_df(step_stats)
        train_losses.index.name = "global_step"
        train_losses.to_csv(os.path.join(args.output_dir, "train_losses.csv"))
