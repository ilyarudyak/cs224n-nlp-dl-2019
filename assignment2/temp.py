import numpy as np

from treebank import StanfordSentiment

dataset = StanfordSentiment()

nTokens = len(dataset.tokens())
samplingFreq = np.zeros((nTokens,))
dataset.allSentences()
i = 0
for w in range(nTokens):
    w = dataset._revtokens[i]
    if w in dataset._tokenfreq:
        freq = 1.0 * dataset._tokenfreq[w]
        # Reweigh
        freq = freq ** 0.75
    else:
        freq = 0.0
    samplingFreq[i] = freq
    i += 1

samplingFreq /= np.sum(samplingFreq)
samplingFreq = np.cumsum(samplingFreq) * dataset.tablesize

dataset._sampleTable = [0] * dataset.tablesize

j = 0
for i in range(dataset.tablesize):
    while i > samplingFreq[j]:
        j += 1
    dataset._sampleTable[i] = j