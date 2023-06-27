import tqdm
from transformers import pipeline
from datasets import load_dataset


classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# load the dataset
dataset = load_dataset("asun17904/bank_examples")
candidate_labels = ["finance", "river"]

def classify_example(example, candidate_labels):
    labels = []
    results = classifier(example["soi"], candidate_labels)
    for result in [v["scores"] for v in results]:
        labels.append(candidate_labels[result.index(max(result))])
    example["labels"] = labels
    return example


dataset = dataset.map(lambda x: classify_example(x, candidate_labels), batched=True)
dataset.push_to_hub("asun17904/bank_examples_with_labels")