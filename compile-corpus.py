#!/usr/bin/env python

import os
import csv
import textwrap
import hashlib
import re

from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser


@dataclass
class Arguments:
    corpus_dir: str = field(
        default="./corpus",
        metadata={"help": "directory containing text files"},
    )

    filename: str = field(
        default="corpus.txt",
        metadata={"help": "name of output file"},
    )

    crap_filename: str = field(
        default="crap.txt",
        metadata={"help": "name of crap output file"},
    )

    format: Optional[str] = field(
        default=None,
        metadata={
            "help": "overrides output format",
            "choices": {"txt", "csv"},
        },
    )

    split: bool = field(
        default=False,
        metadata={"help": "split long paragraphs into multiple entries"},
    )

    split_len: int = field(
        default=512,
        metadata={"help": "max size of split entries"},
    )

    preserve_dupes: bool = field(
        default=False,
        metadata={"help": "preserve duplicate entries"},
    )

    preserve_ws: bool = field(
        default=False,
        metadata={"help": "preserve whitespace"},
    )

    min_len: int = field(
        default=1,
        metadata={"help": "minimum required length of entry"},
    )

    def __post_init__(self):
        if self.format is None:
            self.format = self.filename.split(".")[-1]


parser = HfArgumentParser(Arguments)
(args,) = parser.parse_args_into_dataclasses()

out_fp = open(args.filename, "w", encoding="utf-8")
crap_fp = open(args.crap_filename, "w", encoding="utf-8")

tw = textwrap.TextWrapper(
    width=args.split_len,
    break_long_words=False,
    replace_whitespace=False,
    drop_whitespace=False,
)

processed = 0
entries = 0
total = 0
files = 0
max_len = 0
min_len = 10000
num_split = 0
min_removed = 0
crap_removed = 0
hashes = set()
dupes = 0

if args.format == "csv":
    writer = csv.writer(out_fp, escapechar="\\")
    writer.writerow(["id", "text"])

for filename in os.listdir(args.corpus_dir):
    if not filename.endswith(".txt"):
        continue

    print(filename)
    files = files + 1

    with open(os.path.join(args.corpus_dir, filename), "r", encoding="utf-8") as f:
        text = f.read()

    lines = text.splitlines()

    for line in lines:
        processed = processed + 1

        line = line.replace("\ufeff", "")

        if not args.preserve_ws:
            line = line.strip()
            line = re.sub(r"\s+", " ", line)

        # equations
        if "{\\" in line:
            crap_removed = crap_removed + 1
            #crap_fp.write(f"{line}\n")
            continue

        # must be at least 75% "esperanto-ish"
        tmp1 = re.sub("[A-Za-z ]+", "", line)
        tmp2 = re.sub("\s", "", line)
        if len(tmp1) * 4 > len(tmp2):
            crap_removed = crap_removed + 1
            crap_fp.write(f"{line}\n")
            continue

        if args.split:
            chunks = tw.wrap(line)
            if len(chunks) > 1:
                num_split = num_split + 1
        else:
            chunks = [line]

        chunks_len = len(chunks)
        entries = entries + max(1, chunks_len)

        if chunks_len == 0:
            min_removed = min_removed + 1
            continue

        for chunk in chunks:
            chunk_len = len(chunk)
            if chunk_len < args.min_len:
                min_removed = min_removed + 1
                continue


            if not args.preserve_dupes:
                hash = hashlib.md5(chunk.encode("utf-8")).hexdigest()
                if hash in hashes:
                    dupes = dupes + 1
                    continue
                hashes.add(hash)

            total = total + 1
            min_len = min(min_len, chunk_len)
            max_len = max(max_len, chunk_len)

            if args.format == "csv":
                writer.writerow([total, chunk])
            else:
                out_fp.write(f"{chunk}\n")

out_fp.close()
crap_fp.close()

print(f"\nCreated {args.filename} as {args.format}")
print(f"\nFiles: {files}")
print(f"Lines processed: {processed}")
if args.split:
    print(f"Lines split: {num_split}")
print(f"Entries processed: {entries}")
if not args.preserve_dupes:
    print(f"Duplicate entries: {dupes}")
print(f"Min length: {min_len}")
print(f"Min removed: {min_removed}")
print(f"Crap removed: {crap_removed}")
print(f"Max length: {max_len}")
print(f"Total entries: {total}")

