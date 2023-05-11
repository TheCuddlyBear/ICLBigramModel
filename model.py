import re

class Model:
    def __init__(self, tokens: list):
        self.tokens =  [i.lower() for i in tokens if not re.match('\W', i)]
        for i in range(len(self.tokens)): # loop through every sentence
            p = self.tokens[i] # create temp variable
            s = "<s>" + p + "</s>" # add the boundaries
            self.tokens[i] = s # put it back in the list
