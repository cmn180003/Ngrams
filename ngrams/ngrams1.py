# Homework 4.1
# Crystal Ngo and Kiara Madeam
# https://github.com/cmn180003/NLP-Portfolio
# https://github.com/kiara-aleecia/nlp-portfolio

import sys
import pathlib
import nltk
from nltk import word_tokenize
import pickle

'''
TO DO (PROGRAM 1):
[X]- create function w/ filename as argument
[X]- read in text and remove newlines
[X]- tokenize text
[X]- create bigrams list w/ nltk
[X]- create unigrams list w/ nltk
[X]- use bigram list to create dictionary of bigrams and counts
    - [‘token1 token2’] -> count
[X]- use unigram list to create dictionary of unigrams and counts
    - [‘token’] -> count
[X]- return unigram/bigram dictionary from the function
- IN MAIN:
    - call function 3 times (english, french, italian)
    - pickle 6 dictionaries
'''

'''
Makes language model
Args:
    filename: name of single language file
Returns: dict, dict
    unigram dictionary and bigram dictionary
'''
def make_language_model(filename: str) -> tuple[dict[str, int], dict[tuple[str, str], int]]:
    
    # Reads file
    rel_path = sys.argv[1] + '/' + filename
    with open(pathlib.Path.cwd().joinpath(rel_path), 'r', encoding='utf8') as f:
        text_in = f.read()

    # tokenize text
    tokens = word_tokenize(text_in)
    # Makes unigrams list
    unigrams = tokens

    # Makes bigrams list
    bigrams = [(unigrams[k], unigrams[k + 1]) for k in range(len(unigrams) - 1)]

    # Makes unigram dictionary {str:int}
    uni_dict = {t:unigrams.count(t) for t in set(unigrams)}
    # Makes bigram dictionary {tuple(str, str):int}
    bi_dict = {b:bigrams.count(b) for b in set(bigrams)}

    return uni_dict, bi_dict

'''
create unigram and bigram dictionaries for each language and pickles them
Args:
    n/a
Returns:
    n/a (prints done message)
'''
def main():
    # make uni/bi dict for each language
    eng_uni, eng_bi = make_language_model('LangId.train.English')
    fr_uni, fr_bi = make_language_model('LangId.train.French')
    it_uni, it_bi = make_language_model('LangId.train.Italian')


    # pickle the uni/bi for all three languages (dict)
    # english
    pickle.dump(eng_uni, open('eng_uni.pickle', 'wb'))
    pickle.dump(eng_bi, open('eng_bi.pickle', 'wb'))

    # french
    pickle.dump(fr_uni, open('fr_uni.pickle', 'wb'))
    pickle.dump(fr_bi, open('fr_bi.pickle', 'wb'))

    # italian
    pickle.dump(it_uni, open('it_uni.pickle', 'wb'))
    pickle.dump(it_bi, open('it_bi.pickle', 'wb'))
    
    print("all done!!")

# Starts program
if __name__ == '__main__':
    # Reads system argument
    if len(sys.argv) < 2:
        print("Usage: python3 ngrams1.py data")
        quit()

main()
    