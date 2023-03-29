#!/usr/bin/env python

import random

from dataclasses import dataclass, field
from transformers import HfArgumentParser


@dataclass
class Arguments:
    corpus_file: str = field(
        default="./corpus.txt",
        metadata={"help": "input corpus file"},
    )

    train_file: str = field(
        default="./train_file.txt",
        metadata={"help": "output training file"},
    )

    test_file: str = field(
        default="./test_file.txt",
        metadata={"help": "output test file"},
    )


parser = HfArgumentParser(Arguments)
(args,) = parser.parse_args_into_dataclasses()


random.seed(42)

with open(args.corpus_file, "r", encoding="utf-8") as fp:
    data = fp.readlines()

random.shuffle(data)

i = (int)(len(data) * 0.2)

train_data = data[i:]
with open(args.train_file, "w", encoding="utf-8") as fp:
    fp.write("".join(train_data))

test_data = data[:i]
with open(args.test_file, "w", encoding="utf-8") as fp:
    fp.write("".join(test_data))

print(f"\n{args.train_file}: {len(train_data)}")
print(f"{args.test_file}: {len(test_data)}")



