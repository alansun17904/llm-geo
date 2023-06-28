import os
import nltk
from datasets import load_dataset
from huggingface_hub import HfApi, Repository
from transformers import AutoTokenizer



def separate_sentences(example):
    """
    :example (named tuple): a single example from the dataset with the key "text"
    :return (named tuple): the same example with the key "sentences" added which is
    a list of all of the sentences in the example
    """
    sentences = nltk.sent_tokenize(example["text"])
    example["sentences"] = sentences
    # remove the column "text" from the example
    return example


def extract_target_word_sentences(example, target_word, case_sensitive=False):
    """
    :example (named tuple): a single example from the dataset object with the key "text"
    :return (named tuple): the same example with the key "soi" and "other" added
    the former corresponds to the sentences where the word `target_word` appears and the latter
    corresponds to all of the other sentences in the text.
    """
    sentences = separate_sentences(example)["sentences"]
    soi = []
    other = []
    for i, sentence in enumerate(sentences):
        if case_sensitive:
            if target_word in sentence:
                soi.append(sentence)
            else:
                other.append(sentence)
        elif target_word in sentence.lower():
            soi.append(sentence)
        else:
            other.append(sentence)
    example["soi"] = soi
    example["other"] = other
    return example


if __name__ == '__main__':
    # assumes that the user has defined the environment variable HF_TOKEN
    token = os.environ["HF_TOKEN"]
    # download nltk packages
    nltk.download("punkt")

    # load a partial version of the dataset
    wiki = load_dataset("wikipedia", "20220301.simple")

    # get only the examples with the word "bank"
    bank_examples = wiki["train"].filter(lambda x: "bank" in x["text"].lower())
    no_bank = wiki["train"].filter(lambda x: "bank" not in x["text"].lower())
    no_bank = wiki["train"].map(lambda x: separate_sentences(x))
    # get only the sentences with the word bank
    bank_examples = bank_examples.map(lambda x: extract_target_word_sentences(x, "bank"))

    # remove the column "text" from the bank examples since already separated into sentences
    bank_examples = bank_examples.remove_columns("text")
    
    # check the total number of examples
    print(f"Total number of `bank` sentences: {sum([len(x['soi']) for x in bank_examples])}")

    # push the dataset to the hub
    bank_examples.push_to_hub("asun17904/bank_examples", token=token)
    no_bank.push_to_hub("asun17904/no_bank_examples", token=token)

