import numpy as np

indices = np.load("data/w2v256_0_index2word.npy")
embeddings = []
for i in range(0,21):
    embedding = np.load(f"data/w2v256_{i}.npy")
    embeddings.append(embedding)
embeddings = np.array(embeddings)
print(f"Aggregate shape: {embeddings.shape}")
# export the aggregate into a new file
np.save("data/aggregate256.npy", embeddings)
np.save("data/indices256.npy", indices)

