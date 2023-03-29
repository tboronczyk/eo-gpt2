#!/usr/bin/env python

import os

from dataclasses import dataclass, field
from transformers import HfArgumentParser

import wikipedia


@dataclass
class Arguments:
    output_dir: str = field(
        default="./corpus",
        metadata={"help": "ouput directory"},
    )

    page_list: str = field(
        default="./wikipedia-featured.txt",
        metadata={"help": "file listing featured pages"},
    )


parser = HfArgumentParser(Arguments)
(args,) = parser.parse_args_into_dataclasses()

if not os.path.exists(args.output_dir):
    print(f"Creating output directory: {args.output_dir}")
    os.makedirs(args.output_dir)

wikipedia.set_lang("eo")

with open(args.page_list, "r", encoding="utf-8") as fp:
    pages = [line.strip() for line in fp.readlines()]

total = 0
for page in pages:
    print(page)
    p = wikipedia.page(page)
    with open(
        os.path.join(args.output_dir, f"{page}.txt"), "w", encoding="utf-8"
    ) as fp:
        fp.write(p.content)
    total = total + 1

print(f"\nDownloaded {total} files")

