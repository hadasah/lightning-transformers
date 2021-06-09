from argparse import ArgumentParser
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import transformers
from datasets import Dataset
from lightning_transformers.core.nlp import HFBackboneConfig
from lightning_transformers.core.nlp.seq2seq import Seq2SeqTransformer, Seq2SeqDataConfig, Seq2SeqDataModule
from lightning_transformers.task.nlp.summarization.metric import RougeMetric
from lightning_transformers.prompt_tuning.c4 import *
from lightning_transformers.prompt_tuning.superglue import *
from transformers import default_data_collator, AutoTokenizer, PreTrainedTokenizerBase
from transformers.optimization import Adafactor
import pytorch_lightning as pl


class T5PromptTransformer(Seq2SeqTransformer):

    def __init__(self, *args, downstream_model_type: str = 'transformers.AutoModelForSeq2SeqLM', **kwargs) -> None:
        super().__init__(*args, downstream_model_type, **kwargs)
# ​
#         print(self.model)  # this is the loaded pre-trained model from HF, you can modify it how you want here
# ​
#         # Freeze BERT backbone (assuming bert which is annoying), can just freeze everything then turn requires_grad on for your layer
#         for param in self.model.bert.parameters():
#             param.requires_grad = False

    def configure_optimizers(self) -> Dict:
        """Prepare optimizer and scheduler"""
        optimizer = Adafactor(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=1e-4,
            # eps=(1e-30, 1e-3),
            # clip_threshold=1.0,
            # decay_rate=-0.8, ## TODO change
            # beta1=None,
            weight_decay=1e-5,
            scale_parameter=False,
            relative_step=False,
            # warmup_init=False,
        )
        num_training_steps = self.num_training_steps
        print(num_training_steps)
        num_training_steps, num_warmup_steps = self.compute_warmup(
            num_training_steps,
            # num_warmup_steps=0.1  # will use 10% of the training steps for warmup
            num_warmup_steps=0 # will use 10% of the training steps for warmup #TODO change
        )
        num_warmup_steps = 0
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            },
        }

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        pass

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        pass

    def compute_generate_metrics(self, batch, prefix):
        tgt_lns = self.tokenize_labels(batch["labels"])
        pred_lns = self.generate(batch["input_ids"], batch["attention_mask"])
        result = self.rouge(pred_lns, tgt_lns)
        self.log_dict(result, on_step=False, on_epoch=True)


    def configure_metrics(self, stage: str):
        self.rouge = RougeMetric(
            rouge_newline_sep='\n',
            use_stemmer=False,
        )
    # @property
    # def hf_pipeline_task(self) -> str:
    #     return "summarization"

def add_flags():
    parser = ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default='xsum')
    parser.add_argument('-s', '--subtask', type=str, default=None)
    return parser.parse_args()

def main(task, subtask=None):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='t5-small', local_files_only=True)
    model = T5PromptTransformer(
        backbone=HFBackboneConfig(pretrained_model_name_or_path='t5-small'),
        enc_prompt_length=20,
        # dec_prompt_length=20,
        trunc_from_end=False,
    )
    dm = SuperGlueDataModule(
        cfg=SuperGlueDataConfig(
            batch_size=4,
            # instead of these you could pass dataset_name='csv' 
            # and then specify train_file/validation_file for example.
            dataset_name=task,
            dataset_config_name=subtask,
            num_workers=8,
            # split=['train[:1%]', 'test[:1%]'],
        ),
        tokenizer=tokenizer
    )
    trainer = pl.Trainer(gpus=1, max_epochs=1)

    trainer.fit(model, dm)

if __name__ == '__main__':
    flags = add_flags()
    main(flags.task, flags.subtask)
