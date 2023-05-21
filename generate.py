import nltk

from corpusreader import CorpusReader
from model import BigramModel
from tqdm import tqdm
import sys
def generate_sentence(model: BigramModel, smoothing_constant=0.0):
    sent = ['<s>']
    while sent[len(sent) - 1] != '</s>':
        successor = model.choose_successor(sent[len(sent) - 1], smoothing_constant=smoothing_constant)
        sent.append(successor)
    return sent

def main():
    argument = sys.argv[1]

    if argument is None:
        argument = './train'

    reader = CorpusReader(argument)
    model = BigramModel(reader.sents())

    sents = []
    perplexSents = [
        'Suggestive, Watson, is it not?',
        "It is amazing that a family can be torn apart by something as simple as a pack of wild dogs!",
        "So spoke Sherlock Holmes and turned back to the great scrapbook in which he was arranging and indexing some of his recent material.",
        "What I like best about my friends is that they are few.",
        "Friends what is like are they about I best few my that."
    ]

    print("Generating sentences...")

    for i in range(2):
        print(f"Generating sentence {i}")
        sent = generate_sentence(model)
        sents.append(sent)

    for index, sent in enumerate(sents):
        string = f'Sentence {index}: '
        sent.remove("<s>")
        sent.remove("</s>")
        for x in sent:
            string = string + ' ' + x
        print(string)

    print("Now calculating the perplexities")

    for index, sent in enumerate(perplexSents):
        tokenized = nltk.word_tokenize(sent)
        perplexity = model.perplexity(tokenized, 1.0)
        print(f'The sentence "{sent}" has a perplexity of: {perplexity}')

if __name__ == "__main__":
    main()
