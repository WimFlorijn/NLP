import math
from nltk.tokenize import RegexpTokenizer
from nltk import bigrams, trigrams

import os

class Tokenizer:
    def __init__(self):
        self.tokenizer = RegexpTokenizer(r'\w+')

    def tokenize(self, text, tokenizer=None):
        if tokenizer is None:
            result =  self.tokenizer.tokenize(text)
        else:
            result = tokenizer.tokenize(text)
        return result

    def tokenizeFolder(self, location, gender='All'):
        tokens = []
        for file in os.listdir(location):
            filename = os.fsdecode(file)
            if gender=='All' or filename.startswith(gender):
                with open(location + filename, 'r', encoding='utf-8') as file:
                    content = file.read()
                    tokens.extend(self.tokenize(content))
        return tokens

if __name__ == "__main__":

    #Question 4
    print ('Question 4')
    tokenizer = Tokenizer()

    text1 = 'I am so blue I am greener than purple.'
    text2 = 'I stepped on a Corn Flake, now I am a Cereal Killer'
    text3 = 'I like the course Natural Language Processing'

    print('Tokenize: ' + text1 + ' -> ' + str(tokenizer.tokenize(text1)))
    print('Tokenize: ' + text2 + ' -> ' + str(tokenizer.tokenize(text2)))
    print('Tokenize: ' + text3 + ' -> ' + str(tokenizer.tokenize(text3)))

    print ('\n')

    #Question 5
    print ('Question 5')

    data_dir = 'dataset/blogs/'
    train, test =  'train/', 'test/'
    train_location = data_dir + train

    unigrams = tokenizer.tokenizeFolder(train_location)
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

    unigrams_male = tokenizer.tokenizeFolder(train_location, gender='M')
    unigrams_male_filtered = [i for i in unigrams_male if i in subset_plus25]
    nm = len(unigrams_male_filtered)
    malecount = Counter(unigrams_male_filtered)

    unigrams_female = tokenizer.tokenizeFolder(train_location, gender='F')
    unigrams_female_filtered = [i for i in unigrams_female if i in subset_plus25]
    nf = len(unigrams_female_filtered)
    femalecount = Counter(unigrams_female_filtered)

    test_location = data_dir + test
    k,v = 1,len(set(unigrams_male) | set(unigrams_female))
    print('K: '+ str(k) + ' V: ' + str(v) + ' Nm: ' + str(nm) + ' Nf: ' + str(nf))

    id_class = {}
    for file in os.listdir(test_location):
        filename = os.fsdecode(file)
        with open(test_location + filename, 'r', encoding='utf-8') as file:
            tokens = tokenizer.tokenize(file.read())
            tmale, tfemale = float(0), float(0)
            for token in tokens:
                tmale += math.log(((malecount[token] + k)/(nm + (k*v))), 2)
                tfemale += math.log(((femalecount[token] + k)/(nf + (k*v))), 2)
            if tmale > tfemale:
                id_class[filename[:-4]] = 'M'
            else:
                id_class[filename[:-4]] = 'F'

    with open(data_dir + 'output.txt', 'w') as of:
        for id in id_class:
            of.write(id + '\t' + id_class[id] + '\n')
        of.close()

    import evaluate







