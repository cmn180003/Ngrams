# Homework 4.1
# Crystal Ngo and

import sys
import pathlib
import nltk
from nltk import word_tokenize
import re


def makelangmodel(filename):
    '''
    Makes language model
    Args:
        filename: name of single language file
    Returns: dict, dict
        unigram dictionary and bigram dictionary
    '''
    # Reads file and removes newlines
    rel_path = sys.argv[1] + '/' + filename
    with open(pathlib.Path.cwd().joinpath(rel_path), 'r') as f:
        text_in = f.read()

    # Makes unigrams list
    unigrams = word_tokenize(text_in)
    #print(unigrams)

    # Makes bigrams list
    bigrams = [(unigrams[k], unigrams[k + 1]) for k in range(len(unigrams) - 1)]
    #print(bigrams)

    # Makes unigram dictionary
    uni_dict = {}
    # Makes bigram dictionary
    bi_dict = {}

    return uni_dict, bi_dict

# Starts program
if __name__ == '__main__':
    # Reads system argument
    if len(sys.argv) < 2:
        print('Please enter a filename as a system arg')
        quit()

    # Reads data
    #rel_path = sys.argv[1]

    #Makes language model for each language file
    eng_uni, eng_bi = makelangmodel('LangId.train.English')
    #fr_uni, fr_bi = makelangmodel(LangId.train.French)
    #it_uni, it_bi = makelangmodel(LangId.train.Italian)