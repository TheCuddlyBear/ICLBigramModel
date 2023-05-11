import os

class CorpusReader:
    """Read the contents of a directory of files, and return the results as
    either a list of lines or a list of words.

    The pathname of the directory to read should be passed when
    creating the class:

    >>> reader = CorpusReader(r"path/to/dir")
    """
    def __init__(self, directory):
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
        return wordstring
    
    def words(self) -> list:
        wordstring = self._get_all_text()
        words = wordstring.split()
        return words

    def lines(self) -> list:
        wordstring = self._get_all_text()
        lines = wordstring.splitlines()
        return lines