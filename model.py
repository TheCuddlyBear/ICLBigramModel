import math
import re
import pandas as pd
from collections import Counter
from tqdm import tqdm


class BigramModel:
    def __init__(self, tokens: list):
        if (tokens == None):
            print("Tokens cannot be null!")
        else:
            self.tokens: list = BigramModel.add_sentence_boundaries(tokens)
            _unigram_counts = BigramModel.make_count_unigrams(self.tokens)
            _unigram_counts_tuples = _unigram_counts.most_common(len(_unigram_counts))
            self.unigram_frequency_table = pd.DataFrame(_unigram_counts_tuples, columns=['unigram', 'count'])
            self.unigram_frequency_table.drop(
                self.unigram_frequency_table[self.unigram_frequency_table['unigram'] == '</s>'].index, inplace=True)
            _bigram_counts = BigramModel.make_count_bigrams(self.tokens)
            _bigram_count_tuples = _bigram_counts.most_common(len(_bigram_counts))
            self.bigram_frequency_table = pd.DataFrame(_bigram_count_tuples, columns=['bigram', 'count'])

    def probability(self, w: str, w_n: str, smoothing_constant: float = 0.0):
        """
        @param w: The token we have just seen
        @param w_n: the probability of seeing token w_n
        @param smoothing_constant: the constant with which smoothing is applied
        This function calculates the probability of seeing token w_n after seeing token w
        """
        bigram: tuple = (w, w_n)
        try:
            bigram_count = \
            self.bigram_frequency_table.loc[self.bigram_frequency_table['bigram'] == bigram]['count'].tolist()[0]
        except:
            return 0.0
        try:
            unigram_count = \
            self.unigram_frequency_table.loc[self.unigram_frequency_table['unigram'] == w_n]['count'].tolist()[0]
        except:
            return 0.0
        if smoothing_constant == 0.0:
            # Locate the bigram or unigram we want the probability of
            return bigram_count / unigram_count
        else:
            total_words = len(self.unigram_frequency_table)
            t = bigram_count + smoothing_constant
            n = unigram_count + smoothing_constant * total_words
            return t / n

    def perplexity(self, sent: list, smoothing_constant: float = 1.0) -> float:
        sentCopy = sent.copy()
        sentCopy.remove('</s>')
        n = len(sentCopy)
        probs = []
        for i in range(n - 1):
            probability = self.probability(sentCopy[i], sentCopy[i + 1], smoothing_constant)
            q = 1 / probability
            probs.append(q)
        s = math.prod(probs)
        return s ** (1 / n)

    def choose_successor(self, word: str, smoothing_constant: float = 0.0) -> str | None:
        pass

    @staticmethod
    def make_count_unigrams(tokens: list) -> Counter:
        """
        @param tokens: list of tokenized sentences
        Takes a list of tokenized sentences and generates the appropriate unigrams and counts them
        """
        unigram_counts = Counter()
        for p, words in enumerate(
                tqdm(tokens, ncols=100, desc='Making and counting Unigrams')):  # tqdm prints a progressbar
            unigram_counts += Counter(words)
        return unigram_counts

    @staticmethod
    def make_count_bigrams(tokens: list) -> Counter:
        """
        @param tokens: list of tokenized sentences
        Takes a list of tokenized sentences and generates the appropriate bigrams and counts them
        """
        bigram_counts = Counter()
        for p, words in enumerate(
                tqdm(tokens, ncols=100, desc='Making and counting Bigrams')):  # tqdm prints a progressbar
            bigrams: list = []
            for i in range(len(words) - 1):
                bigrams.append((words[i], words[i + 1]))
            bigram_counts += Counter(bigrams)
        return bigram_counts

    @staticmethod
    def add_sentence_boundaries(tokens: list) -> list:
        """
        @param tokens: list of tokenized sentences
        Takes a list of tokenized sentences and adds sentence boundaries to all the sentences
        """
        tokens_without_punctuation = BigramModel.remove_punctuation_tokens(tokens)
        tokens_with_boundaries: list = []
        item: list
        for i, item in enumerate(
                tqdm(tokens_without_punctuation, ncols=100, desc='Adding boundaries')):  # tqdm prints a progressbar
            item.insert(0, "<s>")
            item.append("</s>")
            tokens_with_boundaries.append(item)
        return tokens_with_boundaries

    @staticmethod
    def remove_punctuation_tokens(tokens: list):
        """
        @param tokens: list of tokenized sentences
        Takes a list of tokenized sentences and removes the tokens that solemnly consist of punctuation from the list
        """
        to_return: list = []
        for sent in tokens:
            to_return.append([p.lower() for p in sent if not re.match('\W', p)])
        return to_return
