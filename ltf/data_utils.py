import numpy as np

BATCH_SIZE = 64
EMBEDDING_DIMENSION = 5
NEGATIVE_SAMPLES = 8
LOG_DIR = "logs/word2vec_intro"
VOCABULARY_SIZE = 9


def get_skip_gram_data():

    digit_to_word_map = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
                         6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}
    sentences = []

    # Create two kinds of sentences - sequences of odd and even digits.
    # ['Five Nine One',
    #  'Eight Six Two',
    #  'Five Three Seven',
    #  'Eight Six Four', ...]
    np.random.seed(42)
    for i in range(10000):
        rand_odd_ints = np.random.choice(range(1, 10, 2), 3)
        sentences.append(" ".join([digit_to_word_map[r] for r in rand_odd_ints]))
        rand_even_ints = np.random.choice(range(2, 10, 2), 3)
        sentences.append(" ".join([digit_to_word_map[r] for r in rand_even_ints]))

    # Map words to indices
    # {'five': 0,
    #  'nine': 1,
    #  'one': 2,
    #  'eight': 3,
    #  'six': 4,
    #  'two': 5,
    #  'three': 6,
    #  'seven': 7,
    #  'four': 8}
    word2index_map = {}
    index = 0
    for sent in sentences:
        for word in sent.lower().split():
            if word not in word2index_map:
                word2index_map[word] = index
                index += 1
    index2word_map = {index: word for word, index in word2index_map.items()}

    vocabulary_size = len(index2word_map)

    # Generate skip-gram pairs
    # [[1, 0], that's ['Nine', 'Five']
    #  [1, 2], that's ['Nine', 'One'] and so on
    #  [4, 3],
    #  [4, 5], .. ]
    skip_gram_pairs = []
    for sent in sentences:
        tokenized_sent = sent.lower().split()
        for i in range(1, len(tokenized_sent)-1):
            word_context_pair = [[word2index_map[tokenized_sent[i-1]],
                                  word2index_map[tokenized_sent[i+1]]],
                                 word2index_map[tokenized_sent[i]]]
            skip_gram_pairs.append([word_context_pair[1],
                                    word_context_pair[0][0]])
            skip_gram_pairs.append([word_context_pair[1],
                                    word_context_pair[0][1]])

    return skip_gram_pairs, word2index_map, index2word_map


def get_skipgram_batch(batch_size, skip_gram_pairs):
    """
    Split pair in 2 parts: x and y. Why are they doing this?
    Batch is randomly chosen from skip_gram_pairs.
    :param batch_size:
    :param skip_gram_pairs:
    :return: batch of input data and labels
    """
    instance_indices = list(range(len(skip_gram_pairs)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    x = [skip_gram_pairs[i][0] for i in batch]
    y = [[skip_gram_pairs[i][1]] for i in batch]
    return x, y
