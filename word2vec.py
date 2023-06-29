import re
import nltk
import json
import tqdm
import numpy as np
from gensim.models import Word2Vec
from datasets import load_dataset
from nltk.corpus import stopwords


def extract_sentences_from_ds(dataset, key, separate=False):
    sentences = []
    for s in tqdm.tqdm(dataset[key], desc=f"Extracting sentences from column `{key}`"):
        sentences += s if not separate else nltk.sent_tokenize(s)
    return sentences


# load all the sentences
def load_initial_sentences(
    no_target_dataset, target_dataset, ratio=0.1, candidate_labels=None
):
    total_target_sentences = sum(
        [
            len(example["soi"])
            for example in target_dataset
            if example["labels"] in candidate_labels
        ]
    )
    print(f"Found a total of {total_target_sentences} target sentences!")
    threshold_sentences = int(total_target_sentences * ratio)
    count = 0
    sentences = extract_sentences_from_ds(no_target_dataset, "sentences")
    sentences += extract_sentences_from_ds(target_dataset, "other")

    # get all of the sentences that do have candidate labels
    for example in target_dataset:
        if count < threshold_sentences and example["labels"] in candidate_labels:
            sentences.extend(example["soi"])
            count += len(example["soi"])
    return sentences


def inject_sentences(target_dataset, ratio=0.1, prev_ratio=0, candidate_labels=None):
    filtered_target = target_dataset.filter(lambda x: x["labels"] in candidate_labels)
    total_target_sentences = len(filtered_target)
    print(f"Found a total of {total_target_sentences} target sentences!")
    prev_sentences = int(total_target_sentences * prev_ratio)
    threshold_sentences = int(total_target_sentences * ratio)
    return extract_sentences_from_ds(
        filtered_target[prev_sentences:threshold_sentences], "soi"
    )


# remove numbers and special characters
def process_text(text):
    text = re.sub("[^A-Za-z]+", " ", text)
    # also remove stop words
    tokens = nltk.word_tokenize(text)
    tokens = [
        w.lower().strip() for w in tokens if w not in stopwords.words("english")
    ]
    return tokens


def train_word2vec_model(sentences, sname, prev_model=None, **kwargs):
    if prev_model is None:
        model = Word2Vec(sentences, **kwargs)
    else:
        model = prev_model
        # fine tune on the new sentences
        model.train(sentences, epochs=1, total_examples=len(sentences))
    # save the model
    model.save(sname)
    # export the vocabulary along with the vectors to a npy file
    vecs = np.asarray(model.wv.vectors)
    index2word = np.asarray(model.wv.index_to_key)
    return model, vecs, index2word


class Sentences:
    def __init__(self, sentences):
        self.sentences = sentences

    def __iter__(self):
        for sentence in self.sentences:
            yield process_text(sentence)

    def __len__(self):
        return len(self.sentences)


if __name__ == "__main__":
    nltk.download('stopwords')
    traj = np.linspace(0.1, 1, 20)
    target_dataset = load_dataset("asun17904/wikitext_bank_examples_with_labels", split="train")
    no_target_dataset = load_dataset("asun17904/wikitext_no_bank_examples", split="train")
    initial_sentences = load_initial_sentences(
        no_target_dataset, target_dataset, ratio=0, candidate_labels=["river"]
    )
    # store all of the initial sentences
    json.dump(initial_sentences, open("initial_sentences.json", "w+"))
    # train the first word2vec model
    model, vecs, index2word = train_word2vec_model(
        Sentences(initial_sentences), "w2v_0.model", vector_size=64,
        workers=32
    )
    np.save(f"w2v_0.npy", vecs)
    np.save(f"w2v_0_index2word.npy", index2word)
    for i, ratio in enumerate(traj):
        sentences = inject_sentences(
            target_dataset,
            ratio=ratio,
            prev_ratio=0 if i == 0 else traj[i - 1],
            candidate_labels=["river"],
        )
        model, vecs, index2word = train_word2vec_model(
            Sentences(sentences),
            f"w2v_{i+1}.model",
            prev_model=model,
            workers=32
        )
        np.save(f"w2v_{i+1}.npy", vecs)
        np.save(f"w2v_{i+1}_index2word.npy", index2word)
