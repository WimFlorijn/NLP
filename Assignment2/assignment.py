from tokenizer import Tokenizer
from collections import Counter
import math, tokenizer, os, nltk


def question_4(custom_tokenizer):
    # Question 4
    print ('Question 4')

    text1 = 'I am so blue I am greener than purple.'
    text2 = 'I stepped on a Corn Flake, now I am a Cereal Killer'
    text3 = 'I like the course Natural Language Processing'

    print('Tokenize: ' + text1 + ' -> ' + str(custom_tokenizer.tokenize(text1)))
    print('Tokenize: ' + text2 + ' -> ' + str(custom_tokenizer.tokenize(text2)))
    print('Tokenize: ' + text3 + ' -> ' + str(custom_tokenizer.tokenize(text3)))

    print ('\n')


def question_5(custom_tokenizer, train_location):
    # Question 5
    print ('Question 5')

    unigrams = custom_tokenizer.tokenize_folder(train_location)
    uunigrams = set(unigrams)
    print('Amount of unigrams: ' + str(len(unigrams)))
    print('Amount of unique unigrams: ' + str(len(uunigrams)))

    bigrams = list(nltk.bigrams(unigrams))
    ubigrams = set(bigrams)
    print('Amount of bigrams: ' + str(len(bigrams)))
    print('Amount of unique bigrams: ' + str(len(ubigrams)))

    trigrams = list(nltk.trigrams(unigrams))
    utrigrams = set(trigrams)
    print('Amount of trigrams: ' + str(len(trigrams)))
    print('Amount of unique trigrams: ' + str(len(utrigrams)))

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
    return subset_plus25


def question_6(custom_tokenizer, data_dir, train_location, subset_plus25):
    # Question 6
    print ('Question 6')

    unigrams_male = custom_tokenizer.tokenize_folder(train_location, gender='M')
    unigrams_male_filtered = [i for i in unigrams_male if i in subset_plus25]
    nm = len(unigrams_male_filtered)
    malecount = Counter(unigrams_male_filtered)

    unigrams_female = custom_tokenizer.tokenize_folder(train_location, gender='F')
    unigrams_female_filtered = [i for i in unigrams_female if i in subset_plus25]
    nf = len(unigrams_female_filtered)
    femalecount = Counter(unigrams_female_filtered)

    test_location = data_dir + test
    k,v = 1,len(set(unigrams_male) | set(unigrams_female))
    print('K: '+ str(k) + ' V: ' + str(v) + ' Nm: ' + str(nm) + ' Nf: ' + str(nf))

    id_class = {}
    r_mtokenindex, r_fmtokenindex = set(), set()
    for file in os.listdir(test_location):
        filename = os.fsdecode(file)
        with open(test_location + filename, 'r', encoding='utf-8') as file:
            tokens = custom_tokenizer.tokenize(file.read())
            tmale, tfemale = float(0), float(0)
            for token in tokens:
                male_score = math.log(((malecount[token] + k)/(nm + (k*v))), 2)
                female_score = math.log(((femalecount[token] + k)/(nf + (k*v))), 2)
                tmale += male_score
                tfemale += female_score
                r_mtokenindex.add((token, (tmale/tfemale)))
                r_fmtokenindex.add((token, (tfemale/tmale)))
            if tmale > tfemale:
                id_class[filename[:-4]] = 'M'
            else:
                id_class[filename[:-4]] = 'F'

    with open(data_dir + 'output.txt', 'w') as of:
        for id in id_class:
            of.write(id + '\t' + id_class[id] + '\n')
        of.close()

    return r_mtokenindex, r_fmtokenindex


def question_61(r_mtokenindex, r_fmtokenindex):
    # Question 6.1
    print ('Question 6.1')

    print(sorted(list(r_mtokenindex), key=lambda x: x[1])[-10:])
    print(sorted(list(r_fmtokenindex), key=lambda x: x[1])[-10:])

if __name__ == "__main__":
    regex_tokenizer = Tokenizer()

    data_dir = 'dataset/blogs/'
    train, test = 'train/', 'test/'
    train_location = data_dir + train

    question_4(regex_tokenizer)

    subset_plus25 = question_5(regex_tokenizer, train_location)

    r_mtokenindex, r_fmtokenindex = question_6(regex_tokenizer, data_dir, train_location, subset_plus25)

    import evaluate

    question_61(r_mtokenindex, r_fmtokenindex)

