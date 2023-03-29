#!/usr/bin/env python

from transformers import pipeline, set_seed

from dataclasses import dataclass, field
from transformers import HfArgumentParser


@dataclass
class Arguments:
    text: str = field(
        default="",
        metadata={"help": "prompt text"},
    )

    max_len: int = field(
        default=512,
        metadata={"help": "maximum length to generate"},
    )

    num_seq: int = field(
        default=1,
        metadata={"help": "number of completion sequences to return"},
    )

    tokenizer_dir: str = field(
        default="./tokenizer",
        metadata={"help": "directory containing tokenizer"},
    )

    model_dir: str = field(
        default="./model",
        metadata={"help": "directory containing model"},
    )

parser = HfArgumentParser(Arguments)
(args,) = parser.parse_args_into_dataclasses()

from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_dir)
model = GPT2LMHeadModel.from_pretrained(args.model_dir)

p = pipeline('text-generation', model=model, tokenizer=tokenizer)

set_seed(42)

results = p(args.text, max_length=args.max_len, num_return_sequences=args.num_seq)

if len(results) == 1:
    print(f"\n{results[0]['generated_text']}")
else:
    for id, text in enumerate(results):
        print(f"\n{id+1}> {text['generated_text']}")

