from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from datasets import Dataset
from lightning_transformers.core.nlp.seq2seq import Seq2SeqTransformer, Seq2SeqDataConfig, Seq2SeqDataModule
from transformers import default_data_collator, PreTrainedTokenizerBase

@dataclass
class C4DataConfig(Seq2SeqDataConfig):
    use_taskname: bool = False
    batch_size: int = 32
    num_workers: int = 0
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    train_val_split: Optional[int] = None
    train_file: Optional[str] = None
    test_file: Optional[str] = None
    validation_file: Optional[str] = None
    padding: Union[str, bool] = "max_length"
    truncation: str = "only_first"
    preprocessing_num_workers: int = 8
    load_from_cache_file: bool = True
    cache_dir: Optional[Union[Path, str]] = None
    limit_train_samples: Optional[int] = None
    limit_val_samples: Optional[int] = None
    limit_test_samples: Optional[int] = None
    max_target_length: int = 64
    max_source_length: int = 512
    split: str = None


class C4DataModule(Seq2SeqDataModule):
    cfg: C4DataConfig

    def __init__(self, *args, cfg: C4DataConfig = C4DataConfig(), tokenizer: PreTrainedTokenizerBase, **kwargs) -> None:
        super().__init__(*args, cfg=cfg, tokenizer=tokenizer, **kwargs)

    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        src_text_column_name, tgt_text_column_name = self.source_target_column_names

        convert_to_features = partial(
            self.convert_to_features,
            tokenizer=self.tokenizer,
            padding=self.cfg.padding,
            max_source_length=self.cfg.max_source_length,
            max_target_length=self.cfg.max_target_length,
            src_text_column_name=src_text_column_name,
            tgt_text_column_name=tgt_text_column_name,
            task_name=self.cfg.dataset_name,
            subtask_name=self.cfg.dataset_config_name,
            task_name_str=self.task_name_str,
        )
        dataset = dataset.map(
            convert_to_features,
            batched=True,
            num_proc=self.cfg.preprocessing_num_workers,
            load_from_cache_file=self.cfg.load_from_cache_file,
        )
        cols_to_keep = [x for x in ["input_ids", "attention_mask", "labels"] if x in dataset["train"].features]
        dataset.set_format(columns=cols_to_keep)
        return dataset

    @property
    def source_target_column_names(self) -> Tuple[str, str]:
        return 'context', 'label_text'

    @staticmethod
    def convert_to_features(
        examples: Any,
        tokenizer: PreTrainedTokenizerBase,
        padding: str,
        max_source_length: int,
        max_target_length: int,
        src_text_column_name: str,
        tgt_text_column_name: str,
        task_name: str,
        subtask_name: str,
        task_name_str: str,
    ):
        
        encoded_results = tokenizer.prepare_seq2seq_batch(
            src_texts=examples[src_text_column_name],
            tgt_texts=examples[tgt_text_column_name],
            max_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
        )
        return encoded_results

    @property
    def collate_fn(self) -> Callable:
        return default_data_collator


    def __init__(self, *args, cfg: LanguageModelingDataConfig = LanguageModelingDataConfig(), **kwargs) -> None:
        super().__init__(*args, cfg=cfg, **kwargs)

    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        column_names = dataset["train" if stage == "fit" else "validation"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        tokenize_function = partial(self.tokenize_function, tokenizer=self.tokenizer, text_column_name=text_column_name)

        dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=self.cfg.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=self.cfg.load_from_cache_file,
        )

        convert_to_features = partial(self.convert_to_features, block_size=self.effective_block_size)

        dataset = dataset.map(
            convert_to_features,
            batched=True,
            num_proc=self.cfg.preprocessing_num_workers,
            load_from_cache_file=self.cfg.load_from_cache_file,
        )

        return dataset

    @property
    def effective_block_size(self) -> int:
        if self.cfg.block_size is None:
            block_size = self.tokenizer.model_max_length
            if block_size > 1024:
                log.warn(
                    f"The tokenizer picked seems to have a very large `model_max_length` "
                    f"({self.tokenizer.model_max_length}). "
                    "Picking 1024 instead. You can change that default value by passing dataset.cfg.block_size=x."
                )
            block_size = 1024
        else:
            if self.cfg.block_size > self.tokenizer.model_max_length:
                log.warn(
                    f"The block_size passed ({self.cfg.block_size}) is larger than the maximum length for the model"
                    f"({self.tokenizer.model_max_length}). Using block_size={self.tokenizer.model_max_length}."
                )
            block_size = min(self.cfg.block_size, self.tokenizer.model_max_length)
        return block_size

    @staticmethod
    def tokenize_function(
        examples,
        tokenizer: Union[PreTrainedTokenizerBase],
        text_column_name: str = None,
    ):
        return tokenizer(examples[text_column_name])

    @staticmethod
    def convert_to_features(examples, block_size: int, **kwargs):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    @property
    def collate_fn(self) -> Callable:
        return default_data_collator
