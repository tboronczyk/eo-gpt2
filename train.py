#!/usr/bin/env python

import os
from transformers import AutoTokenizer
from transformers import TextDataset,DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM

from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser


@dataclass
class Arguments:
    tokenizer: str = field(
        default="./tokenizer",
        metadata={"help": "tokenizer to use"},
    )

    model_dir: str = field(
        default="./model",
        metadata={"help": "directory containing model files"},
    )

    train_file: str = field(
        default="./train_file.txt",
        metadata={"help": "training data file"},
    )

    test_file: str = field(
        default="./test_file.txt",
        metadata={"help": "test data file"},
    )

#    checkpoint: Optional[int] = field(
#        default=None,
#        metadata={"help": "checkpoint to continue training from"},
#    )

    train_batch_size: int = field(
        default=32,
        metadata={"help": "batch size for training"},
    )

    eval_batch_size: int = field(
        default=64,
        metadata={"help": "batch size for evaluation"},
    )

    eval_steps: int = field(
        default=500,
        metadata={"help": "number of update steps between two evaluations"},
    )

    save_steps: int = field(
        default=1000,
        metadata={"help": "number of steps after which model is saved"},
    )

    init: bool = field(
        default=False,
        metadata={
            "help": "initial train on distilgpt2",
        },
    )

    epocs: int = field(
        default=3,
        metadata={
            "help": "number of training epocs",
        },
    )

parser = HfArgumentParser(Arguments)
(args,) = parser.parse_args_into_dataclasses()

if args.init:
    init_model = "distilgpt2"
else:
    init_model = args.model_dir

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

def load_dataset(train_path, test_path, tokenizer):
    train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=128)

    test_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=test_path,
          block_size=128)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    return train_dataset, test_dataset, data_collator

train_dataset, test_dataset, data_collator = load_dataset(args.train_file, args.test_file, tokenizer)

model = AutoModelForCausalLM.from_pretrained(init_model)

training_args = TrainingArguments(
    output_dir=args.model_dir,
    overwrite_output_dir=True,
    num_train_epochs=args.epocs,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    eval_steps = args.eval_steps,
    save_steps=args.eval_steps,
    warmup_steps=500,
    optim="adamw_torch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

trainer.save_model()
