from corpusreader import CorpusReader
from model import BigramModel
import sys

def main():
    argument = sys.argv[1]

    if argument is None:
        argument = './train'

    reader = CorpusReader(argument)
    model = BigramModel(reader.sents())

    def generate_sentence(model: BigramModel, smoothing_constant=0.0):
        sent = ['<s>']
        while sent[len(sent) - 1] != '</s>':
            successor = model.choose_successor(sent[len(sent) - 1], smoothing_constant=smoothing_constant)
            sent.append(successor)
        return sent

    sent = generate_sentence(model, 1.0)

    string = ''
    for i in sent:
        string = string + ' ' + i
    print(string)

if __name__ == "__main__":
    main()
