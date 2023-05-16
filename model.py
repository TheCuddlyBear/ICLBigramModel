import re
import pandas as pd
from collections import Counter
import nltk

class BigramModel:
    def __init__(self, tokens: list):
        self.tokens: list = self.addSentenceBoundaries([i.lower() for i in tokens if not re.match('\W', i)], True)
        self.frequencyTable: pd.DataFrame = pd.DataFrame(data={'bigram': [], 'count': []})
        bigramCounts = self.countBigrams(self.tokens)
        print("Bigram counting done!")
        bigramCountTuples = bigramCounts.most_common(len(bigramCounts))
        self.frequencyTable: pd.DataFrame = pd.DataFrame(bigramCountTuples, columns=['bigram', 'count'])
    
    def countBigrams(self, tokens: list) -> Counter:
        bigramCounts = Counter()
        l = len(tokens)
        self.printProgressBar(0, l, prefix='Counting Bigrams:', suffix= 'Complete', length = 50)
        for i, sentence in enumerate(tokens):
            bigrams: list = []
            words: list = sentence.split(" ")
            words.remove("</s>")
            for i in range(len(words) - 1):
                bigrams.append(words[i] + " " + words[i+1])
            bigramCounts += Counter(bigrams)
            self.printProgressBar(i+1, l, prefix='Counting Bigrams:', suffix= 'Complete', length = 50)
        return bigramCounts

    def addSentenceBoundaries(self, tokens: list, useProgressBar=False) -> list:
        tokensWithBoundaries: list = []
        if useProgressBar:
            l = len(tokens)
            self.printProgressBar(0, l, prefix = 'adding boundaries:', suffix = 'Complete', length = 50)
        for i in range(len(tokens)):
            s = "<s> " + tokens[i] + " </s>"
            tokensWithBoundaries.append(s)
            if useProgressBar:
                self.printProgressBar(i+1, l, prefix = 'adding boundaries:', suffix = 'Complete', length = 50)
        return tokensWithBoundaries

    def printProgressBar(self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        # Print New Line on Complete
        if iteration == total: 
            print()