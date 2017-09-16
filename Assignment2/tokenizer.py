import nltk
from nltk.tokenize import RegexpTokenizer
from nltk import bigrams, trigrams

import os

class Tokenizer:
    def __init__(self, location):
        self.location = location

    @staticmethod
    def tokenize(text):
        tokens = RegexpTokenizer(r'\w+').tokenize(text)
        return tokens

    @staticmethod
    def tokenizeFolder(location):
        tokens = []
        for file in os.listdir(location):
            filename = os.fsdecode(file)
            if filename.endswith(".txt"):
                with open(location + filename, 'r', encoding='utf-8') as file:
                    content = file.read()
                    tokens.extend(Tokenizer.tokenize(content))
        return tokens


if __name__ == "__main__":

    #Question 4
    print ('Question 4')

    text1 = 'I am so blue I am greener than purple.'
    text2 = 'I stepped on a Corn Flake, now I am a Cereal Killer'
    text3 = 'I like the course Natural Language Processing'

    print('Tokenize: ' + text1 + ' -> ' + str(Tokenizer.tokenize(text1)))
    print('Tokenize: ' + text2 + ' -> ' + str(Tokenizer.tokenize(text2)))
    print('Tokenize: ' + text3 + ' -> ' + str(Tokenizer.tokenize(text3)))

    print ('\n')

    #Question 5
    print ('Question 5')

    data_dir = 'dataset/blogs/'
    train =  'train/'
    location = data_dir + train

    unigrams = Tokenizer.tokenizeFolder(location)
    uunigrams = set(unigrams)
    print('Amount of unigrams: ' + str(len(unigrams)))
    print('Amount of unique unigrams: ' + str(len(uunigrams)))
    bigrams = list(bigrams(unigrams))
    ubigrams = set(bigrams)
    print('Amount of bigrams: ' + str(len(bigrams)))
    print('Amount of unique bigrams: ' + str(len(ubigrams)))
    trigrams = list(trigrams(unigrams))
    utrigrams = set(trigrams)
    print('Amount of trigrams: ' + str(len(trigrams)))
    print('Amount of unique trigrams: ' + str(len(utrigrams)))

    from collections import Counter
    count = Counter(unigrams)
    common = count.most_common()

    for i in range(0,3):
        print('Most common place ' + str((i+1)) + ' word: \'' + common[i][0] + '\', frequency: ' + str(common[i][1]))

    #Retrieve all ocurance values for each word and make a counter class again
    occ = Counter(count.values())
    for i in range(1,5):
        print('Amount of words which occur ' + str(i) + ' times: ' + str(occ[i]))
    print ('\n')

    #Question 6





