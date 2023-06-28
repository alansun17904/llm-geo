import nltk
import numpy as np
from dataset import separate_sentences
from gensim.models import Word2Vec
from datasets import load_dataset


# load all the sentences
def load_sentences(no_target_dataset, target_dataset, ratio=0.1, candidate_labels=None):
    no_target_dataset = load_dataset(no_target_dataset)
    target_dataset = load_dataset(target_dataset)
    total_target_sentences = sum([
        sum([len(example[v] for v  in candidate_labels)]) for example in target_dataset["train"]
    ])
    threshold_sentences = int(total_target_sentences * ratio)
    count = 0
    sentences = []
    for example in no_target_dataset["train"]:
        sentences.extend(separate_sentences(example)["text"])
    for example in target_dataset["train"]:
        sentences.extend(example["other"])
    for example in target_dataset["train"]:
        for target_word in candidate_labels:
            sentences.extend(example[target_word])
            count += len(example[target_word])
        if count > threshold_sentences:
            break
    return sentences


def train_word2vec_model(sentences, sname, **kwargs):
    model = Word2Vec(sentences, **kwargs)
    # save the model
    model.save(sname)
    # export the vocabulary along with the vectors to a npy file
    vecs = np.asarray(model.wv.vectors)
    index2word = np.asarray(model.wv.index_to_key)
    return vecs, index2word


if __name__ == '__main__':
    traj = np.linspace(0.1, 1, 20)
    for ratio in traj:
        sentences = load_sentences(
            "asun17904/no_bank_examples",
            "asun17904/bank_examples_with_labels",
            ratio=ratio,
            candidate_labels=["river"]
        )
        vecs, index2word = train_word2vec_model(
            sentences, f"w2v_{ratio}.model", size=64, window=5, min_count=1, workers=4
        )
        np.save(f"w2v_{ratio}.npy", vecs)
        np.save(f"w2v_{ratio}_index2word.npy", index2word)
    


