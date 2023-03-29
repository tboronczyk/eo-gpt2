#!/usr/bin/env python

from transformers import AutoTokenizer

from dataclasses import dataclass, field
from transformers import HfArgumentParser


@dataclass
class Arguments:
    tokenizer_dir: str = field(
        default="./tokenizer",
        metadata={"help": "directory containing tokenizer files"},
    )

    train_file: str = field(
        default="./train_file.txt",
        metadata={"help": "training data file"},
    )

    vocab_size: str = field(
        default=6500,
        metadata={"help": "vocabulary size"},
    )

    init: bool = field(
        default=False,
        metadata={
            "help": "initial train on distilgpt2 tokenizer",
        },
    )


parser = HfArgumentParser(Arguments)
(args,) = parser.parse_args_into_dataclasses()

if args.init:
    init_tokenizer = "distilgpt2"
else:
    init_tokenizer = args.tokenizer_dir

def corpus_iterator(train_file):
    with open(train_file, "r", encoding="utf-8") as fp:
        lines = fp.readlines()
    return (text for text in lines)


corpus = corpus_iterator(args.train_file)

old_tokenizer = AutoTokenizer.from_pretrained(init_tokenizer)

tokenizer = old_tokenizer.train_new_from_iterator(corpus, args.vocab_size)

tokenizer.save_pretrained(args.tokenizer_dir)
