#!/usr/bin/env python

import os
import urllib.request
import zipfile

from dataclasses import dataclass, field
from transformers import HfArgumentParser

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager


@dataclass
class Arguments:
    output_dir: str = field(
        default="./corpus",
        metadata={"help": "ouput directory"},
    )

    tmp_dir: str = field(
        default="./tmp",
        metadata={"help": "temporary directory"},
    )


parser = HfArgumentParser(Arguments)
(args,) = parser.parse_args_into_dataclasses()

if not os.path.exists(args.output_dir):
    print(f"Creating output directory: {args.output_dir}")
    os.makedirs(args.output_dir)

if not os.path.exists(args.tmp_dir):
    print(f"Creating tmp directory: {args.tmp_dir}")
    os.makedirs(args.tmp_dir)

zip_path = os.path.join(args.tmp_dir, "tekstaro.zip")
if not os.path.exists(zip_path):
    print(f"Downloading tekstaro.zip... (please wait)")
    urllib.request.urlretrieve(
        "https://tekstaro.com/elshutebla/tekstaro_de_esperanto_html_sen_streketoj.zip",
        zip_path,
    )

text_dir = os.path.join(args.tmp_dir, "tekstaro_de_esperanto_html_sen_streketoj", "tekstoj")
if not os.path.exists(text_dir):
    with zipfile.ZipFile(zip_path, "r") as fp:
        fp.extractall(args.tmp_dir)

files = []
for filename in os.listdir(text_dir):
    if not filename.endswith(".html"):
        continue
    files.append(filename)

files.remove("homaranismo-1906.html")

options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--disable-web-security")

driver = webdriver.Chrome(
    service=ChromeService(ChromeDriverManager(cache_valid_range=7).install()), options=options
)

for file in files:
    print(file)
    url = "file://" + os.path.realpath(os.path.join(text_dir, file))
    driver.get(url)

    for sel in ["body h2:first-of-type", "#alaktualasekcio", ".tekstokapo", ".noto", ".m"]:
        driver.execute_script(f"document.querySelectorAll('{sel}').forEach(el => el.remove())")

    text = driver.find_element(By.XPATH, "/html/body").text

    with open(os.path.join(args.output_dir, file.split('.')[0]+".txt"), "w", encoding="utf-8") as fp:
        fp.write(text)

driver.quit()

print(f"\nCopied {len(files)} files")

