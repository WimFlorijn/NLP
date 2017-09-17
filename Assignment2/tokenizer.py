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
    def tokenizeFolder(location, gender='All'):
        tokens = []
        for file in os.listdir(location):
            filename = os.fsdecode(file)
            if filename.endswith(".txt") and (gender=='All' or filename.startswith(gender)):
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
    test = 'test/'
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

    subset_plus25 = []
    for item in count.keys():
        if count[item] > 24:
            subset_plus25.append(item)

    print ('\n')

    #Question 6
    print ('Question 6')

    import math
    unigrams_male = Tokenizer.tokenizeFolder(location, gender='M')
    unigrams_male_filtered = [i for i in unigrams_male if i in subset_plus25]
    totalmale = len(unigrams_male_filtered)
    malecount = Counter(unigrams_male_filtered)
    print (malecount)

    unigrams_female = Tokenizer.tokenizeFolder(location, gender='F')
    unigrams_female_filtered = [i for i in unigrams_female if i in subset_plus25]
    totalfemale = len(unigrams_female_filtered)
    femalecount = Counter(unigrams_female_filtered)
    print(femalecount)

    location = data_dir + test
    k=5
    v = len(set(unigrams_male) | set(unigrams_female))

    fileclassification = {}
    for file in os.listdir(location):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            with open(location + filename, 'r', encoding='utf-8') as file:
                content = file.read()
                tokens = Tokenizer.tokenize(content)
                tmale = float(0)
                tfemale = float(0)
                for token in tokens:
                    tmale += math.log(((malecount[token] + k)/(totalmale + (k*v))), 2)
                    tfemale += math.log(((femalecount[token] + k)/(totalfemale + (k*v))), 2)
                if tmale > tfemale:
                    fileclassification[filename[:-4]] = 'M'
                else:
                    fileclassification[filename[:-4]] = 'F'

    with open(data_dir + 'output.txt', 'w') as of:
        for filename in fileclassification:
            of.write(filename + '\t' + fileclassification[filename] + '\n')
        of.close()







