import re
import pandas as pd
from collections import Counter
from tqdm import tqdm


class BigramModel:
    def __init__(self, tokens: list):
        if (tokens == None):
            print("Tokens cannot be null!")
        else:
            self.tokens: list = self.__add_sentence_boundaries(tokens)
            _bigram_counts = self.__make_count_bigrams(self.tokens)
            _bigram_count_tuples = _bigram_counts.most_common(len(_bigram_counts))
            self.frequencyTable: pd.DataFrame = pd.DataFrame(_bigram_count_tuples, columns=['bigram', 'count'])

    def probability(self, w: str, w_n: str, smooth_constant: float = 0.0) -> float:
        pass

    def perplexity(self, sent: list, smoothing_constant: float = 0.0) -> float:
        pass

    def choose_successor(self, word: str, smooth_constant: float = 0.0) -> str:
        pass

    def __make_count_bigrams(self, tokens: list) -> Counter:
        bigram_counts = Counter()
        for p, words in enumerate(tqdm(tokens, ncols=100, desc='Making and counting Bigrams')):
            bigrams: list = []
            for i in range(len(words) - 1):
                bigrams.append(words[i] + " " + words[i + 1])
            bigram_counts += Counter(bigrams)
        return bigram_counts

    def __add_sentence_boundaries(self, tokens: list) -> list:
        tokens_without_punctuation = self.__remove_punctuation_tokens(tokens)
        tokens_with_boundaries: list = []
        item: list
        for i, item in enumerate(tqdm(tokens_without_punctuation, ncols=100, desc='Adding boundaries')):
            item.insert(0, "<s>")
            item.append("</s>")
            tokens_with_boundaries.append(item)
        return tokens_with_boundaries

    def __remove_punctuation_tokens(self, tokens: list):
        to_return: list = []
        for sent in tokens:
            to_return.append([p.lower() for p in sent if not re.match('\W', p)])
        return to_return