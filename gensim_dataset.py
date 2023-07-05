import os
import re
import nltk
import gensim.downloader as api
from datasets import Dataset


nltk.download("punkt")
dataset = api.load("wiki-english-20171001")
token = os.environ["HF_TOKEN"]
bank_match = re.compile(r"\b(?:bank)\b", re.IGNORECASE)

def gen():
    for d in dataset:
        yield d


def check_bank(sentence):
    return any([
        re.search(bank_match, s) is not None 
        for s in sentence
    ])

def separate_nested_sentences(example):
    s = []
    for sentence in example["section_texts"]:
        s.extend(nltk.sent_tokenize(sentence))
    example["sentences"] = s
    return example

def get_sentences_of_interest(example):
    soi = []
    other = []
    for s in example["sentences"]:
        if re.search(bank_match, s) is None:
            other.append(s)
        else:
            soi.append(s)
    example["soi"] = soi
    example["other"] = other
    return example


dataset = Dataset.from_generator(gen)
bank_examples = dataset.filter(lambda x: check_bank(x["section_texts"]))
no_bank_examples = dataset.filter(lambda x: not check_bank(x["section_texts"]))

# separate the sentences and then only keep those that match the regex again
bank_examples = bank_examples.map(lambda x: separate_nested_sentences(x))

# re-get the sentences that have the word of interest in them. 
bank_examples = bank_examples.map(lambda x: get_sentences_of_interest(x))

# remove the sentence column
bank_examples = bank_examples.remove_columns("sentences")
print(
    f"Total number of `bank` sentences: {sum([len(x['soi']) for x in bank_examples])}"
)

# push both datasets to the hub
bank_examples.push_to_hub("asun17904/wiki2017_bank_examples", token=token)
no_bank_examples.push_to_hub("asun17904/wiki2017_no_bank_examples", token=token)

