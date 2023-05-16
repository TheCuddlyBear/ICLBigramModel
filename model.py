import re
import pandas as pd
from collections import Counter
from tqdm import tqdm


class BigramModel:
    def __init__(self, tokens: list):
        self.tokens: list = self.addSentenceBoundaries(tokens)
        bigramCounts = self.countBigrams(self.tokens)
        bigramCountTuples = bigramCounts.most_common(len(bigramCounts))
        self.frequencyTable: pd.DataFrame = pd.DataFrame(bigramCountTuples, columns=['bigram', 'count'])

    def countBigrams(self, tokens: list) -> Counter:
        bigramCounts = Counter()
        for p, words in enumerate(tqdm(tokens, ncols=100, desc='Making and counting Bigrams')):
            bigrams: list = []
            for i in range(len(words) - 1):
                bigrams.append(words[i] + " " + words[i + 1])
            bigramCounts += Counter(bigrams)
        return bigramCounts

    def addSentenceBoundaries(self, tokens: list) -> list:
        tokensWithoutPunctuation = self.removePunctuationTokens(tokens)
        tokensWithBoundaries: list = []
        item: list
        for i, item in enumerate(tqdm(tokensWithoutPunctuation, ncols=100, desc='Adding boundaries')):
            item.insert(0, "<s>")
            item.append("</s>")
            tokensWithBoundaries.append(item)
        return tokensWithBoundaries

    def removePunctuationTokens(self, tokens: list):
        toReturn: list = []
        for sent in tokens:
            toReturn.append([p.lower() for p in sent if not re.match('\W', p)])
        return toReturn