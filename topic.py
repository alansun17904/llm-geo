import os
import tqdm
from transformers import pipeline
from datasets import load_dataset

import datasets

datasets.logging.set_verbosity_error()

classifier = pipeline(
    "zero-shot-classification", model="facebook/bart-large-mnli", device=0
)
# load the dataset
dataset = load_dataset("asun17904/bank_examples")
candidate_labels = ["money", "riverbed"]


def classify_example(example, candidate_labels):
    labels = []
    results = classifier(example["soi"], candidate_labels)
    for result in [v["labels"] for v in results]:
        labels.append(result[0])
    example["labels"] = labels
    return example


dataset = dataset.map(lambda x: classify_example(x, candidate_labels), batched=True)
token = os.environ["HF_TOKEN"]
dataset.push_to_hub("asun17904/bank_examples_with_labels", token=token)
