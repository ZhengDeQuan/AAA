import numpy as np
import tensorflow as tf
import pickle
import os

vocab = None
vocab_size = 10
embedding_size = 8
embedding_matrix = np.random.uniform(-1, 1, size=(vocab_size, embedding_size))

# for w, i in word_index.items():
#     v = embeddings.get(w)
#     if v is not None and i < vocab_size:
#         embedding_matrix[i] = v

# for i in range(vocab_size):
#     embedding_matrix[i] = np.ones(embedding_size) * i * 2

def my_initializer1(shape=None, dtype=tf.float32, partition_info=None):
    assert dtype is tf.float32
    return embedding_matrix

def tagset2vocab(tagset):
    global embedding_matrix
    if os.path.exists("embedding_matrix.pkl"):
        embedding_matrix = pickle.load(open("embedding_matrix.pkl","rb"))
        return

    global vocab,vocab_size,embedding_size
    vocab = tagset
    vocab_size = len(vocab) + 1
    pretrained_embedding_dict = pickle.load(open("videoID_vector.pkl","rb"))
    for key in pretrained_embedding_dict:
        embedding_size=len(pretrained_embedding_dict[key]['vector'])
        break
    embedding_matrix = np.random.uniform(-1, 1, size=(vocab_size, embedding_size))
    for idx, word in enumerate( vocab ):
        if word in pretrained_embedding_dict:
            vector = pretrained_embedding_dict[word]["vector"]
            embedding_matrix[idx] = vector
    pickle.dump(embedding_matrix,open("embedding_matrix.pkl","wb"))

def fun1(message):
    print(message)
    return np.random.uniform(-1, 1, size=(vocab_size, embedding_size))

custom_tags = [
{'tag_name':'7','vocab_size':130000,'embedding_size':768,'initializer_function':my_initializer1,'vocab_fun':tagset2vocab}
]

custom_tags = []