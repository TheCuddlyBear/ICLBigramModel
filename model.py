import re
from collections import Counter
import pandas as pd
import nltk


class Model:
    def __init__(self, tokens: list):
        self.tokens: list =  [i.lower() for i in tokens if not re.match('\W', i)]
        self.frequency: pd.DataFrame = pd.DataFrame(data={'bigram': [], 'count': []})
        for i in range(len(self.tokens)): # loop through every sentence
            p = self.tokens[i] # create temp variable
            s = "<s> " + p + " </s>" # add the boundaries
            self.tokens[i] = s # put it back in the list
        sentence: str
        for sentence in self.tokens:
            bigrams: list = []
            words: list = sentence.split(" ")
            words.remove("</s>")
            for i in range(len(words) - 1):
                bigrams.append(words[i] + " " + words[i+1])
            



    def printTable(self):
        print(self.frequency.info())
            
            

        
    
