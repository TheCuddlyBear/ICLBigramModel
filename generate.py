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

    for i in tqdm(range(2), ncols=100, desc="Generating sentences"):
        sent = generate_sentence(model)
        sents.append(sent)


    for i in sents:
        string = ''
        for x in i:
            string = string + ' ' + x
        print(string)

if __name__ == "__main__":
    main()
