import itertools
import numpy as np


def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): list of distinct words across the corpus, sorted (using python 'sorted' function)
            num_corpus_words (integer): number of distinct words across the corpus
    """
    corpus_words = []
    num_corpus_words = -1

    # ------------------
    # Write your implementation here.
    corpus_words = set(itertools.chain.from_iterable(corpus))
    corpus_words = sorted(corpus_words)
    num_corpus_words = len(corpus_words)
    # ------------------

    return corpus_words, num_corpus_words


def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).

        Note: Each word in a document should be at the center of a window. Words near edges will have a smaller
              number of co-occurring words.

              For example, if we take the document "START All that glitters is not gold END" with window size of 4,
              "All" will co-occur with "START", "that", "glitters", "is", and "not".

        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (numpy matrix of shape (number of corpus words, number of corpus words)):
                Co-occurence matrix of word counts.
                The ordering of the words in the rows/columns should be the same as the ordering of the words given by the distinct_words function.
            word2Ind (dict): dictionary that maps word to index (i.e. row/column number) for matrix M.
    """
    words, num_words = distinct_words(corpus)
    M = np.zeros((num_words, num_words))
    word2Ind = {word: i for i, word in enumerate(words)}

    # ------------------
    # Write your implementation here.
    for article in corpus:
        count_words_in_article(article, word2Ind, M, window_size)
    # ------------------

    return M, word2Ind


def count_words_in_article(article, word2Ind, M, window_size=1):
    for i, _ in enumerate(article):
        mi = word2Ind[article[i]]
        # print(f'i={i} mi={mi} word={article[i]}')
        for j in range(i - window_size, i + window_size + 1):
            if 0 <= j < len(article):
                # print(f'j={j}', end=' ')
                mj = word2Ind[article[j]]
                # print(f'mj={mj}, word={article[j]}', end=' ')
                if mi != mj:
                    M[mi, mj] += 1
        # print('\n')


if __name__ == '__main__':
    test_corpus = ["START All that glitters isn't gold END".split(" "),
                   "START All's well that ends well END".split(" ")]
    M_test, word2Ind_test = compute_co_occurrence_matrix(test_corpus, window_size=1)

    word2Ind_ans = {'All': 0, "All's": 1, 'END': 2, 'START': 3, 'ends': 4,
                    'glitters': 5, 'gold': 6, "isn't": 7,
                    'that': 8, 'well': 9}

    M_test_ans = np.array(
        [[0., 0., 0., 1., 0., 0., 0., 0., 1., 0., ],
         [0., 0., 0., 1., 0., 0., 0., 0., 0., 1., ],
         [0., 0., 0., 0., 0., 0., 1., 0., 0., 1., ],
         [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., ],
         [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., ],
         [0., 0., 0., 0., 0., 0., 0., 1., 1., 0., ],
         [0., 0., 1., 0., 0., 0., 0., 1., 0., 0., ],
         [0., 0., 0., 0., 0., 1., 1., 0., 0., 0., ],
         [1., 0., 0., 0., 1., 1., 0., 0., 0., 1., ],
         [0., 1., 1., 0., 1., 0., 0., 0., 1., 0., ]]
    )

    print(np.allclose(M_test, M_test_ans))
