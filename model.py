import math
import pickle
import re
import pandas as pd
from collections import Counter
from tqdm import tqdm
import random

class BigramModel:
    def __init__(self, tokens: list):
        if tokens is None:
            print("Tokens cannot be null!")
        else:
            self.tokens: list = BigramModel.add_sentence_boundaries(tokens)
            _unigram_counts = BigramModel.count_unigrams(self.tokens)
            _unigram_counts_tuples = _unigram_counts.most_common(len(_unigram_counts))
            self.unigrams = pd.DataFrame(_unigram_counts_tuples, columns=['unigram', 'count'])
            #self.unigrams.drop(self.unigrams.loc[self.unigrams['unigram'] == '</s>'].index, inplace=True)
            _bigram_counts = BigramModel.make_count_bigrams(self.tokens)
            _bigram_count_tuples = _bigram_counts.most_common(len(_bigram_counts))
            self.bigrams = pd.DataFrame(_bigram_count_tuples, columns=['bigram', 'count'])

    def __setstate__(self, state):
        self.unigrams = pd.DataFrame.from_dict(state['unigram_pickle'])
        self.bigrams = pd.DataFrame.from_dict(state['bigram_pickle'])
        self.tokens = state['tokens']

    def __getstate__(self):
        dict = {'bigram_pickle': self.bigrams.to_dict(),
                'unigram_pickle': self.unigrams.to_dict(), 'tokens': self.tokens}
        return dict

    def probability(self, w: str, w_n: str, smoothing_constant: float = 0.0):
        """
        @param w: The token we have just seen
        @param w_n: the probability of seeing token w_n
        @param smoothing_constant: the constant with which smoothing is applied
        This function calculates the probability of seeing token w_n after seeing token w
        """
        bigram: tuple = (w, w_n)
        bigram_count = self.bigrams.loc[self.bigrams['bigram'] == bigram]['count'].tolist()
        unigram_count = self.unigrams.loc[self.unigrams['unigram'] == w_n]['count'].tolist()

        if smoothing_constant == 0.0:
            # Locate the bigram or unigram we want the probability of
            if len(bigram_count) != 0 and len(unigram_count) !=0:
                return bigram_count[0] / unigram_count[0]
            else:
                return 0.0
        else:
            total_words = len(self.unigrams)

            if len(bigram_count) == 0 and len(unigram_count) == 0:
                bigram_count = 0.0
                unigram_count = 0.0
                t = bigram_count + smoothing_constant
                n = unigram_count + smoothing_constant * total_words
                return t / n
            elif len(bigram_count) != 0 and len(unigram_count) != 0:
                t = bigram_count[0] + smoothing_constant
                n = unigram_count[0] + smoothing_constant * total_words
                return t / n


    def perplexity(self, sent: list, smoothing_constant: float = 1.0) -> float:
        """
        @param sent: Sentence in the form of a list of tokens
        @param smoothing_constant: The constant used to apply smoothing
        This calculates the perplexity of the given sentence.
        """
        if '<s>' not in sent:
            sent.insert(0, '<s>')
            sent.append('</s>')
            words = [p.lower() for p in sent if not re.match('\W', p)]
        n = len(sent)
        probs = []
        for i in range(n - 1):
            probability = math.log(self.probability(sent[i], sent[i + 1], smoothing_constant))
            q = -probability
            probs.append(q)
        s = sum(probs)
        return math.exp(s * (1/n))

    def choose_successor(self, word: str, smoothing_constant: float = 0.0) -> str | None:
        """
        @param word: The word/token for which to generate the successor
        @param smoothing_constant: The constant used to apply smoothing
        This function uses the probability function to randomly but based on probabilities
        choose a successor to the token given.
        """
        try:
            unigram_count = \
                self.unigrams.loc[self.unigrams['unigram'] == word]['count'].tolist()[0]
        except:
            return None

        possible_bigrams = \
            self.bigrams[self.bigrams['bigram'].apply(lambda x: x[0] == word)][
                'bigram'].tolist()
        prob2 = []
        for bigram in tqdm(possible_bigrams, ncols=100, desc=f"Choosing successor for: {word}"):
            prob2.append(self.probability(word, bigram[1], 0.0))
        successor: tuple = random.choices(possible_bigrams, weights=prob2, k=1)
        return successor[0][1]

    def save_model(self, location: str):
        """
        @param location: Path to where the model should be stored
        Uses pickle to store the model into a file to be loaded back in later
        """
        with open(location, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(location: str):
        """
        @param location: Path to where the model is stored
        Uses to pickle to load a store model back in to e.g. a variable.
        """
        with open(location, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def count_unigrams(tokens: list) -> Counter:
        """
        @param tokens: list of tokenized sentences
        Takes a list of tokenized sentences and generates the appropriate unigrams and counts them
        """
        totWords: list = []
        for p, words in enumerate(
                tqdm(tokens, ncols=100, desc='Making and counting Unigrams')):  # tqdm prints a progressbar
            for word in words:
                totWords.append(word)
        return Counter(totWords)

    @staticmethod
    def make_count_bigrams(tokens: list) -> Counter:
        """
        @param tokens: list of tokenized sentences
        Takes a list of tokenized sentences and generates the appropriate bigrams and counts them
        """
        # bigram_counts = Counter()
        bigrams: list = []
        for p, words in enumerate(tqdm(tokens, ncols=100, desc='Making and counting Bigrams')):  # tqdm prints a progressbar
            #words.remove("</s>")
            for i in range(len(words) - 1):
                bigrams.append((words[i], words[i + 1]))
        return Counter(bigrams)

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
