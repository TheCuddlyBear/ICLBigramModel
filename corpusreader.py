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
        """
        Gets all the text files in the model's path parameter, and puts them into one string.
        """
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
        return wordstring

    def sents(self) -> list:
        """
        Generates a list of sentences which are represented as lists of tokens (words)
        """
        toReturn: list = []
        wordString = self._get_all_text()
        wordsStringOneLine = wordString.replace('\n', ' ').replace('  ', ' ')
        sents = nltk.sent_tokenize(wordsStringOneLine)
        for i in sents:
            words = nltk.word_tokenize(i)
            toReturn.append(words)
        return toReturn
    
    def words(self) -> list:
        """
        Generates a list of tokens (words)
        """
        wordstring = self._get_all_text()
        words = nltk.word_tokenize(wordstring)
        return words

    def lines(self) -> list:
        """
        Generates a list of all the lines
        """
        wordstring = self._get_all_text()
        lines = wordstring.splitlines()
        return lines