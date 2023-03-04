# Homework 4.2
# Crystal Ngo and Kiara Madeam
# https://github.com/cmn180003/NLP-Portfolio
# https://github.com/kiara-aleecia/nlp-portfolio

import sys
import pathlib
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
import pickle

'''
TO DO:
[X]- read in pickled dictionaries
[X]- for each line in test file:
    [X]- calculate probability for each language
    [X]- write language with highest probability to a file
[X]- compute and output accuracy as 'percentage of CORRECTLY CLASSIFIED
    instances in the test set' (check .sol file)
[X]- output your accuracy
[X]- output line numbers of incorrectly classified items
'''

'''
calculate probability for line language based on bigrams/unigrams dict
Args: string, dict{tuple(str, str): int}, dict{str: int}, list[str] 
    line of file, bi dict {bigram: count}, uni dict {uni: count}, hashset of vocab
Returns: double
    probability w/ laplace smoothing
'''
def test_language_model(line: str, bi_dict: dict[tuple[str, str], int], uni_dict: dict[str, int], V: int) -> float:
    unigrams_test = word_tokenize(line)
    bigrams_test = list(ngrams(unigrams_test, 2))
    
    p_laplace = 1  # calculate p using Laplace smoothing

    for bigram in bigrams_test:
        n = bi_dict[bigram] if bigram in bi_dict else 0
        d = uni_dict[bigram[0]] if bigram[0] in uni_dict else 0
        p_laplace = p_laplace * ((n + 1) / (d + V))
    
    return p_laplace
    
'''
Makes language model
Args:
    n/a
Returns: 
    n/a (writes to file with language guesses)
'''
def main():
    filename = 'data/LangId.test'
    # read the pickles back in
    eng_bi = pickle.load(open('eng_bi.pickle', 'rb'))
    eng_uni = pickle.load(open('eng_uni.pickle', 'rb'))

    fr_bi = pickle.load(open('fr_bi.pickle', 'rb'))
    fr_uni = pickle.load(open('fr_uni.pickle', 'rb'))

    it_bi = pickle.load(open('it_bi.pickle', 'rb'))
    it_uni = pickle.load(open('it_uni.pickle', 'rb'))

    # open file to write guesses and read in test file lines
    res_file = open('calculate_language.txt', 'w')
    with open(pathlib.Path.cwd().joinpath(filename), 'r', encoding='utf8') as f:
        text_in = f.read().splitlines()

    # calculate probability for each language and write to file
    vocab = eng_uni | fr_uni | it_uni
    V = len(set(vocab))

    # test probability for each language and write most probable to file
    for line in text_in:
        eng_prob = test_language_model(line, eng_bi, eng_uni, V)
        fr_prob = test_language_model(line, fr_bi, fr_uni, V)
        it_prob = test_language_model(line, it_bi, it_uni, V)
        results = max(eng_prob, fr_prob, it_prob)

        if(results == eng_prob):
            res_file.write('English\n')
        elif(results == fr_prob):
            res_file.write('French\n')
        elif(results == it_prob):
            res_file.write('Italian\n')
    res_file.close()

    # read in file that we just created and wrote our results to
    filename = 'calculate_language.txt'
    with open(pathlib.Path.cwd().joinpath(filename), 'r', encoding='utf8') as f:
        results = f.read().splitlines()
    
    # check against .sol file and output accuracy and lines missed
    filename = 'data/LangId.sol'
    with open(pathlib.Path.cwd().joinpath(filename), 'r', encoding='utf8') as f:
        text_in = f.read().splitlines()
    answers = [tuple(line.split()) for line in text_in]

    missed = []
    for i in range(len(results)):
        key = answers[i]
        if(results[i] != key[1]):
            missed.append(key[0])
    correct = len(results) - len(missed)
    accuracy = (correct / len(results)) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    for m in missed:
        print(f"Incorrect on line {m}")

    

if __name__ == '__main__':
    # Reads system argument
    if len(sys.argv) != 1:
        
        print("Usage: python3 ngrams2.py")
        quit()
     

main()
   