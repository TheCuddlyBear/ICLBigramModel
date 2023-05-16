import os
import nltk

class CorpusReader:
    """Read the contents of a directory of files, and return the results as
    either a list of lines or a list of words.

    The pathname of the directory to read should be passed when
    creating the class:

    >>> reader = CorpusReader(r"path/to/dir")
    """
    def __init__(self, directory: str):
       if os.path.isdir(directory):
           self.path = directory
       else:
           raise ValueError(directory + " does not exist or is not a directory")

    def _get_all_text(self) -> str:
        files = []
        wordstring = ''
        for (dirpath, dirnames, filenames) in os.walk(self.path):
            files.extend(filenames)
            break
        filtered = [i for i in files if i.endswith('txt')]
        for filename in filtered:
            path = self.path + '/' + filename
            with open(path, 'r') as f:
                wordstring += f.read()
        wordsStringOneLine = wordstring.replace('\n', ' ').replace('  ', ' ')
        return wordsStringOneLine

    def sents(self) -> list:
        toReturn: list = []
        wordString = self._get_all_text()
        sents = nltk.sent_tokenize(wordString)
        for i in sents:
            words = nltk.word_tokenize(i)
            toReturn.append(words)
        return toReturn
    
    def words(self) -> list:
        wordstring = self._get_all_text()
        words = nltk.word_tokenize(wordstring)
        return words

    def lines(self) -> list:
        wordstring = self._get_all_text()
        lines = wordstring.splitlines()
        return lines