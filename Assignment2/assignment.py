from tokenizer import Tokenizer
from collections import Counter
import math, os, nltk
from math import log, pow

# Smoothing parameter `k`
K = 1


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


def question_5(custom_tokenizer, train_location, min_occ = 25):
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
        if count[item] >= min_occ:
            subset_plus25.append(item)

    print ('\n')
    return subset_plus25


def question_6(custom_tokenizer, data_dir, train_location, test_location, subset_plus25):
    # Question 6
    print ('Question 6')

    unigrams_male, nm, malecount = _get_unigrams_filtered(
        custom_tokenizer, train_location, subset_plus25, 'M')
    unigrams_female, nf, femalecount = _get_unigrams_filtered(
        custom_tokenizer, train_location, subset_plus25, 'F')

    v = len(set(unigrams_male) | set(unigrams_female))
    print(f'K: {K} V: {v} Nm: {nm} Nf: {nf}')

    id_class = {}
    r_mtokenindex, r_fmtokenindex = set(), set()

    for file in os.listdir(test_location):
        filename = os.fsdecode(file)
        with open(test_location + filename, 'r', encoding='utf-8') as file:
            tokens = custom_tokenizer.tokenize(file.read())
            tmale, tfemale = float(0), float(0)
            for token in tokens:
                male_score = log(((malecount[token] + K)/(nm + (K*v))), 2)
                female_score = log(((femalecount[token] + K)/(nf + (K*v))), 2)
                tmale += male_score
                tfemale += female_score
                rev_male_score, rev_female_score = pow(2, male_score), pow(2, female_score)
                r_mtokenindex.add((token, (rev_male_score/rev_female_score)))
                r_fmtokenindex.add((token, (rev_female_score/rev_male_score)))
            if tmale > tfemale:
                id_class[filename[:-4]] = 'M'
            else:
                id_class[filename[:-4]] = 'F'

    with open(data_dir + 'output6.txt', 'w') as of:
        for id in id_class:
            of.write(id + '\t' + id_class[id] + '\n')
        of.close()
    print('\n')
    return r_mtokenindex, r_fmtokenindex


def question_61(r_mtokenindex, r_fmtokenindex, amount=10):
    # Question 6.1
    print('\n')
    print ('Question 6.1')
    male = sorted(list(r_mtokenindex), key=lambda x: x[1], reverse=True)
    female = sorted(list(r_fmtokenindex), key=lambda x: x[1], reverse=True)
    if len(male) < amount or len(female) < amount:
        amount = min(len(male), len(female))

    top_male = male[:amount]
    top_female = female[:amount]

    print('Top ' + str(amount) + ' male words:')
    print(top_male)
    print('Top ' + str(amount) + ' female words:')
    print(top_female)
    print('\n')


def question_7(tokenizer, data_dir, train_dir, test_dir, subset_plus25):
    # Question 7
    print('Question 7')

    bigrams_m, n_m, count_m = _get_bigrams_filtered(
        tokenizer, train_dir, subset_plus25, 'M')
    bigrams_f, n_f, count_f = _get_bigrams_filtered(
        tokenizer, train_dir, subset_plus25, 'F')
    v = len(set(bigrams_m) | set(bigrams_f))
    print(f'K: {K} V: {v} Nm: {n_m} Nf: {n_f}')

    id_class = {}
    r_mtokenindex, r_fmtokenindex = set(), set()

    for filename_b in os.listdir(test_dir):
        filename = os.fsdecode(filename_b)
        with open(
                os.path.join(test_dir, filename), 'r', encoding='utf-8'
        ) as file:
            bigrams = _get_bigrams_filtered(
                tokenizer, file.read(), subset_plus25, multiple=False)[0]
            bigram = bigrams[0]
            score_m = log((count_m[bigram]+K)/(n_m+K*v), 2)
            score_f = log((count_f[bigram]+K)/(n_f+K*v), 2)
            t_m = score_m
            t_f = score_f
            rev_score_m, rev_score_f = pow(2, score_m), pow(2, score_f)
            r_mtokenindex.add((bigram, rev_score_m/rev_score_f))
            r_fmtokenindex.add((bigram, rev_score_f/rev_score_m))
            for i, bigram in enumerate(bigrams[1:]):
                score_m = log((count_m[bigram]+K)/(count_m[bigrams[i-1]]+K), 2)
                score_f = log((count_f[bigram]+K)/(count_f[bigrams[i-1]]+K), 2)
                t_m += score_m
                t_f += score_f
                rev_score_m, rev_score_f = pow(2, score_m), pow(2, score_f)
                r_mtokenindex.add((bigram, rev_score_m/rev_score_f))
                r_fmtokenindex.add((bigram, rev_score_f/rev_score_m))
            id_class[filename[:-4]] = 'M' if t_m > t_f else 'F'

    with open(os.path.join(data_dir, 'output7.txt'), 'w') as of:
        for id in id_class:
            of.write(id + '\t' + id_class[id] + '\n')
        of.close()

    return r_mtokenindex, r_fmtokenindex


def _get_unigrams_filtered(
        custom_tokenizer, train_location, subset_plus25, gender='All'):
    unigrams = custom_tokenizer.tokenize_folder(train_location, gender=gender)
    unigrams_filtered = [i for i in unigrams if i in subset_plus25]
    n = len(unigrams_filtered)
    count = Counter(unigrams_filtered)
    return unigrams, n, count


def _get_bigrams_filtered(
        tokenizer, source, subset_plus25, gender=None, multiple=True):
    # Extract the normalized tokens from the training set
    if multiple:
        tokens = tokenizer.tokenize_folder(source, gender=gender)
    else:
        tokens = tokenizer.tokenize(source)

    # Get a list of all bigrams from the provided tokens
    # Filter on bigrams containing at least one of the 25+ tokens
    bigrams = list(nltk.bigrams(tokens))
    filtered_bigrams = list(filter(
        lambda b: all(w in subset_plus25 for w in b), bigrams))
    n = len(filtered_bigrams)
    count = Counter(filtered_bigrams)

    return filtered_bigrams, n, count


if __name__ == "__main__":
    regex_tokenizer = Tokenizer()

    data_dir = 'dataset/blogs/'
    train, test = 'train/', 'test/'
    train_location, test_location = data_dir + train, data_dir + test

    question_4(regex_tokenizer)

    # 25 by default, set to 50+ for testing question 6.1
    subset_plus25 = question_5(regex_tokenizer, train_location, min_occ=25)

    r_mtokenindex, r_fmtokenindex = question_6(regex_tokenizer, data_dir, train_location, test_location, subset_plus25)
    question_61(r_mtokenindex, r_fmtokenindex, amount=10)

    r_mtokenindex, r_fmtokenindex = question_7(regex_tokenizer, data_dir, train_location, test_location, subset_plus25)
    question_61(r_mtokenindex, r_fmtokenindex, amount=10)

