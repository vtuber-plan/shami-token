import json

from transformers import AutoTokenizer
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

import os
from datasets import load_dataset

langs = ["en", "ja", "ko", "zh-cn", "zh-tw"]
raw_datasets = [
    load_dataset("wiki40b", lang, beam_runner='DirectRunner')
    for lang in langs
]

total_line = 0
for training_dataset in raw_datasets:
    for line in training_dataset["train"]:
        total_line += 1

def training_dataset_iterator():
    for training_dataset in raw_datasets:
        for line in training_dataset["train"]:
            yield line['text']

# tokenizer.train(training_files, trainer)
tokenizer = old_tokenizer.train_new_from_iterator(training_dataset_iterator(), 102400, total_line)

tokenizer.save_pretrained("tokenizer-shami")
