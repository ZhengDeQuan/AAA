import numpy as np
import tensorflow as tf
import pickle
import os



vocab = None
# vocab_size = 598633
vocab_size = 125080+1
embedding_size = 768
embedding_matrix = np.random.uniform(-1, 1, size=(vocab_size, embedding_size))


embedding_matrix_placeholder = tf.placeholder(np.float32)
embedding_matrix_variable = tf.Variable(embedding_matrix_placeholder)

# for w, i in word_index.items():
#     v = embeddings.get(w)
#     if v is not None and i < vocab_size:
#         embedding_matrix[i] = v

# for i in range(vocab_size):
#     embedding_matrix[i] = np.ones(embedding_size) * i * 2

def my_initializer1(shape=None, dtype=tf.float32, partition_info=None):
    assert dtype is tf.float32
    return embedding_matrix_variable
    # return embedding_matrix_variable
    # W = tf.constant(embedding_matrix, name="W")
    # W = tf.Variable(embedding_matrix,trainable=False, name="W",dtype=tf.float64)
    # return W

def tagset2vocab(tagset):
    global custom_tags
    global embedding_matrix
    # if os.path.exists("embedding_matrix.pkl"):
    #     embedding_matrix = pickle.load(open("embedding_matrix.pkl","rb"))
    #     return

    global vocab,vocab_size,embedding_size
    vocab = list(tagset)
    vocab_size = len(vocab) + 1
    print("in taegset2 vocab")
    print("vocab_size = ",vocab_size)
    # import pdb
    # pdb.set_trace()
    custom_tags[0]['vocab_size']=vocab_size
    pretrained_embedding_dict = pickle.load(open("videoID_vector.pkl.filterComment","rb"))
    for key in pretrained_embedding_dict:
        embedding_size=len(pretrained_embedding_dict[key]['vector'])
        break
    embedding_matrix = np.random.uniform(-1, 1, size=(vocab_size, embedding_size))
    for idx, word in enumerate( vocab ):
        if word in pretrained_embedding_dict:
            vector = pretrained_embedding_dict[word]["vector"]
            embedding_matrix[idx] = vector
    pickle.dump(embedding_matrix,open("embedding_matrix.pkl","wb"))
    global embedding_matrix_variable, embedding_matrix_placeholder
    embedding_matrix_placeholder = tf.placeholder(shape=[vocab_size,embedding_size])
    embedding_matrix_variable = tf.Variable(embedding_matrix_placeholder)
    return vocab


custom_tags = [
{'tag_name':'7','vocab_size':598633,'embedding_size':768,'initializer_function':my_initializer1,'vocab_fun':tagset2vocab}
]
