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

import datasets
import torch
import transformers


def _assert_lengths_attention_mask_col_input_id_col_same_values(lengths):
    assert lengths.attention_mask.eq(
        lengths.input_ids
    ).all(), "Attention mask and input ids have different lengths."


class BeaverTails:
    TRAIN_VAL_SPLIT_SIZES = [24672, 2514]

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
    ):
        """Load and preprocess the beavertails 30k dataset.

        See https://huggingface.co/datasets/PKU-Alignment/BeaverTails.

        Args:
            tokenizer (transformers.PreTrainedTokenizer): HuggingFace pretrained
            tokenizer to tokenize the raw text.

        Raises:
            ValueError: `size` not in ['30k', '330k'].

        Notes:
        - Dataset curation. Prompts from Anthropic HH dataset, fed to open source
        LLMs (alpaca 7b in the case of the 30k dataset) then labelled by crowdworkers
        into 14 categories. An example gets the safe meta label if
        it is safe according to all of the categories.
        - Examples not unique. Prompts repeated, sometimes getting different
        outputs, then sometimes both prompt and output repeated, sometimes getting
        different labels.
        - Beaver tails 30k has one crowd worker (so one annotation?) per QA pair,
        but 330k has on average 3.34.
        """

        self.tokenizer = tokenizer

        raw_full_train_ds = datasets.load_dataset(
            "PKU-Alignment/BeaverTails",
            split=f"30k_train",
        )
        raw_test_ds = datasets.load_dataset(
            "PKU-Alignment/BeaverTails",
            split=f"30k_test",
        )
        self._check_dataset(raw_full_train_ds)
        self._check_dataset(raw_test_ds)

        self._raw_full_train_ds = raw_full_train_ds
        self._raw_test_ds = raw_test_ds

        self.category_names = list(raw_full_train_ds[0]["category"].keys())
        self.raw_ds = datasets.DatasetDict(
            {"train": raw_full_train_ds, "test": raw_test_ds}
        )

    @staticmethod
    def _check_dataset(split):
        df = split.to_pandas()
        assert df.notna().any(axis=None), "Null values"
        assert (
            df.category.map(lambda x: tuple(x.keys())).nunique() == 1
        ), "Category keys vary across examples"

    @staticmethod
    def process_example(
        example: dict[str, str | bool],
        tokenizer: transformers.PreTrainedTokenizer,
        **tokenizer_kwds,
    ) -> dict[str, torch.Tensor]:
        templated = f"user: {example['prompt']}\n\nagent: {example['response']}"
        tokenized = tokenizer(templated, **tokenizer_kwds)
        targets = list(map(int, example["category"].values()))
        return {**tokenized, "targets": targets}

    @property
    def n_categories(self):
        return len(self.category_names)

    def max_tokenized_len(self) -> int:
        """Compute max length of tokenized string in BeaverTails datasets.

        Expected to be downloaded from HuggingFace, see
        https://huggingface.co/datasets/PKU-Alignment/BeaverTails

        Returns:
            int: The max length of tokenized inputs.
        """

        def get_lengths(dataset):
            return (
                dataset.map(
                    lambda x: self.process_example(x, self.tokenizer),
                    remove_columns=dataset.column_names,
                )
                .to_pandas()
                .drop(["targets"], axis=1)
                .map(len)
            )

        train_lengths = get_lengths(self._raw_full_train_ds)
        test_lengths = get_lengths(self._raw_test_ds)

        _assert_lengths_attention_mask_col_input_id_col_same_values(train_lengths)
        _assert_lengths_attention_mask_col_input_id_col_same_values(test_lengths)

        return max(train_lengths.max(axis=None), test_lengths.max(axis=None))

    def train_val_test_splits(self, rng: torch.Generator):
        max_length = self.max_tokenized_len()

        def func(example):
            return self.process_example(
                example,
                self.tokenizer,
                padding="max_length",
                # This will truncate any examples longer than the model max sequence length.
                # See https://huggingface.co/docs/transformers/v4.41.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer
                # for how tokenizer.model_max_length is set.
                max_length=min(max_length, self.tokenizer.model_max_length),
                truncation=True,
            )

        full_train_ds = self._raw_full_train_ds.map(
            func,
            remove_columns=self._raw_full_train_ds.column_names,
        )
        test_ds = self._raw_test_ds.map(
            func, remove_columns=self._raw_test_ds.column_names
        )

        train_ds, val_ds = torch.utils.data.random_split(
            full_train_ds,
            lengths=self.TRAIN_VAL_SPLIT_SIZES,
            generator=rng,
        )
        return train_ds, val_ds, test_ds
