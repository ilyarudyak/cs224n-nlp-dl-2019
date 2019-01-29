import numpy as np
import tensorflow as tf


def generate_data():
    digit_to_word_map = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
                         6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 0: "PAD"}

    even_sentences = []
    odd_sentences = []

    # we keep the original sentence lengths to pass it
    # to tf.nn.dynamic_rnn()
    seqlens = []

    # (1) generate even and odd sentences of different length
    # and pad them to have the same len=6
    # but keep seqlens
    for i in range(10000):
        rand_seq_len = np.random.choice(range(3, 7))
        seqlens.append(rand_seq_len)
        rand_odd_ints = np.random.choice(range(1, 10, 2),
                                         rand_seq_len)
        rand_even_ints = np.random.choice(range(2, 10, 2),
                                          rand_seq_len)

        if rand_seq_len < 6:
            rand_odd_ints = np.append(rand_odd_ints,
                                      [0]*(6-rand_seq_len))
            rand_even_ints = np.append(rand_even_ints,
                                       [0]*(6-rand_seq_len))

        even_sentences.append(" ".join([digit_to_word_map[r] for r in rand_even_ints]))
        odd_sentences.append(" ".join([digit_to_word_map[r] for r in rand_odd_ints]))

    # (2) concatenate data and make one-hot encoding of labels
    data = even_sentences+odd_sentences
    seqlens *= 2
    labels = [1] * 10000 + [0] * 10000
    for i in range(len(labels)):
        label = labels[i]
        one_hot_encoding = [0]*2
        one_hot_encoding[label] = 1
        labels[i] = one_hot_encoding

    # (3) create word-to =index and vise versa
    # why not to use digit_to_word_map?
    # Note that there is no correspondence between the word IDs and the digits
    # each word represents—the IDs carry no semantic meaning.
    word2index_map = {}
    index = 0
    for sent in data:
        for word in sent.lower().split():
            if word not in word2index_map:
                word2index_map[word] = index
                index += 1

    index2word_map = {index: word for word, index in word2index_map.items()}

    # (4) shuffle data
    data_indices = list(range(len(data)))
    np.random.shuffle(data_indices)
    data = np.array(data)[data_indices]
    labels = np.array(labels)[data_indices]
    seqlens = np.array(seqlens)[data_indices]

    # (5) just split in half fot train and test
    train_x = data[:10000]
    train_y = labels[:10000]
    train_seqlens = seqlens[:10000]

    test_x = data[10000:]
    test_y = labels[10000:]
    test_seqlens = seqlens[10000:]

    return train_x, train_y, train_seqlens, test_x, test_y, test_seqlens, word2index_map


def get_sentence_batch(batch_size, data_x,
                       data_y, data_seqlens,
                       word2index_map):
    instance_indices = list(range(len(data_x)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    x = [[word2index_map[word] for word in data_x[i].lower().split()]
         for i in batch]
    y = [data_y[i] for i in batch]
    seqlens = [data_seqlens[i] for i in batch]
    return x, y, seqlens
