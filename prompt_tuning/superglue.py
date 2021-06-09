from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from datasets import Dataset
from lightning_transformers.core.nlp.seq2seq import Seq2SeqTransformer, Seq2SeqDataConfig, Seq2SeqDataModule
from transformers import default_data_collator, PreTrainedTokenizerBase

@dataclass
class SuperGlueDataConfig(Seq2SeqDataConfig):
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
    # max_length: int = 128
    preprocessing_num_workers: int = 8
    load_from_cache_file: bool = True
    cache_dir: Optional[Union[Path, str]] = None
    limit_train_samples: Optional[int] = None
    limit_val_samples: Optional[int] = None
    limit_test_samples: Optional[int] = None
    max_target_length: int = 64
    max_source_length: int = 512
    split: str = None


class SuperGlueDataModule(Seq2SeqDataModule):
    cfg: SuperGlueDataConfig

    def __init__(self, *args, cfg: SuperGlueDataConfig = SuperGlueDataConfig(), tokenizer: PreTrainedTokenizerBase, **kwargs) -> None:
        super().__init__(*args, cfg=cfg, tokenizer=tokenizer, **kwargs)
        self.task_name_str = ''
        if self.cfg.use_taskname:
            self.task_name_str = self.cfg.dataset_name.split('.')[0] + ' '

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
        # return 'document', 'summary'
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
        def make_superglue_seq2seq(
            task_name: str, subtask_name: str, task_name_str: str, examples: Any
        ):
            TF_table = {
                    -1: "",
                    0: "False",
                    1: "True",
                },
            label_lookup = {
                'boolq': TF_table,
                'cb': {
                    -1: "",
                    0: "entailment",
                    1: "contradiction",
                    2: "neutral",
                }, 
                'copa': TF_table,
                'multirc': TF_table,
                'record': {
                },
                'rte': {
                    -1: "",
                    0:"entailment",
                    1:"not_entailment",
                },
                'wic': TF_table,
            }
            label_dict = label_lookup.get(subtask_name)
            if subtask_name == 'boolq':
                examples['context'] = ['{}hypothesis: {} premise: {}'.format(task_name_str, e['hypothesis'], e['premise']) for e in examples]
                raise Error('unimplemented')
            elif subtask_name == 'cb':
                examples['context'] = ['{}hypothesis: {} premise: {}'.format(task_name_str, h, p) for h, p in zip(examples['hypothesis'], examples['premise'])]
            elif subtask_name == 'copa':
                examples['context'] = ['{}choice1: {} choice2: {} question: {}'.format(task_name_str, c1, c2, q) for c1, c2, q in zip(examples['choice1'], examples['choice2'], examples['question'])]
            elif subtask_name == 'multirc':
                examples['context'] = ['{}question: {} answer: {} paragraph {}'.format(task_name_str, q, a, p) for q, a, p in zip()]
            elif subtask_name == 'record':
                raise Error('unimplemented')
            elif subtask_name == 'rte':
                examples['context'] = ['{}sentence1: {} sentence2: {}'.format(task_name_str, s1, s2) for s1,s2 in zip(e['sentence1'], e['sentence2'])]
            elif subtask_name == 'wic':
                examples['context'] = ['{}pos: {} sentence1: {} sentence2: {}'.format(task_name_str, p, s1, s2) for p, s1, s2 in zip()]
            elif subtask_name in ['wsc', 'wsc.fixed']:
                examples['context'] = [''] #needs to be fixed with True labels only
                raise Error('unimplemented')
            else: 
                raise Error('subtask not in set of superglue tasks')
            
            # add T/F labels, NLI labels text to examples
            if subtask_name != ['record', 'wsc', 'wsc.fixed']:
                examples['label_text'] = []
                for l in examples['label']:
                    examples['label_text'].append(label_dict[l])

            print(examples)
            return examples

        if task_name == 'super_glue':
            examples = make_superglue_seq2seq(task_name, subtask_name, task_name_str, examples)
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