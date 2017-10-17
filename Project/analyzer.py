import os
import json
import shutil

from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.util import ngrams

from dataset import TwitterDataSet
from tokenizer import Tokenizer

default_sid = SentimentIntensityAnalyzer()

DEFAULT_CONFIG_DIR = "config"
DEFAULT_INPUT_DATA_DIR = "data"
DEFAULT_OUTPUT_DATA_DIR = "processed_data"
DEFAULT_EXCEPTION_LIST = ['HASH_TAG', 'URL', 'I']


class Analyzer:

    tokenizer = Tokenizer()

    def __init__(
            self,
            config_dir=DEFAULT_CONFIG_DIR,
            data_dir=DEFAULT_INPUT_DATA_DIR,
            output_dir=DEFAULT_OUTPUT_DATA_DIR):
        """
        Initializes the analyzer class with a data set.

        :param data_dir: the directory to which tweets are saved.
        """

        self.config_dir = config_dir
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.exception_list = DEFAULT_EXCEPTION_LIST
        self.tweets = {}
        self.processed_tweets = {}
        self.inverse_index = {}

        # Load the previously downloaded tweets
        self.load()

    def load(self):
        """
        Loads the tweets from the data directory.
        """

        data_set = TwitterDataSet(self.config_dir, self.data_dir)
        self.tweets = data_set.get_preprocessed_tweets()

    def save(self):
        """
        Saves the processed tweets to the output directory.
        """

        # Create the data directory if it exists, then create a new
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.mkdir(self.output_dir)

        users_file = os.path.join(self.config_dir, 'users.txt')

        # Read the list of users from the users.txt file
        with open(users_file) as users:
            elements = {}
            for line in users.readlines():
                if line.lstrip().startswith('#') or len(line.strip()) == 0:
                    continue
                party, user = line.strip().split(':', 1)
                if party not in elements.keys():
                    elements[party] = {}
                if user in self.tweets.keys():
                    elements[party][user] = self.tweets[user]

            self.processed_tweets = elements

            for element_key in elements.keys():
                with open(os.path.join(self.output_dir, element_key + '.json'), 'w') as f:
                    json.dump(elements[element_key], f)

    def add_to_inverse_index(self, tokens):
        for token in tokens:
            if token in self.inverse_index:
                self.inverse_index[token] += 1
            else:
                self.inverse_index[token] = 1

    def calc_features(
            self,
            sentiment_analysis_feature=True,
            n_gram_feature=True,
            count_feature=True,
            punctuation_feature=True):

        for person in self.tweets:
            for tweet in self.tweets[person]:
                tweet_text = self.tweets[person][tweet]['text']
                tokens = self.tokenizer.tokenize(tweet_text)
                if sentiment_analysis_feature:
                    ss = self.calculate_sentiment(tokens)
                    for k in sorted(ss):
                        self.tweets[person][tweet][k] = ss[k]

                if n_gram_feature:
                    for i in range(1, 5):
                        score = 0
                        if i > 1:
                            n_grams = self.calculate_word_grams(tokens, n=i)
                            #n_gram_tokens = list()
                            #for value in n_grams:
                            #    n_gram_tokens.extend(value.split())
                            n_gram_tokens = n_grams
                        else:
                            n_gram_tokens = tokens
                        #for token in n_gram_tokens:
                        #    score += 1 / self.inverse_index[token]
                        self.tweets[person][tweet][str(i)+'_gram'] = n_gram_tokens

                if count_feature:
                    self.tweets[person][tweet]['charc'] = self.calculate_character_count(tweet_text)
                    self.tweets[person][tweet]['wordc'] = self.calculate_word_count(tokens)

                if punctuation_feature:
                    self.tweets[person][tweet]['excc'] = self.calculate_symbol_occ(tweet_text, '!')
                    self.tweets[person][tweet]['quesc'] = self.calculate_symbol_occ(tweet_text, '?')
                    self.tweets[person][tweet]['quotec'] = self.calculate_symbol_occ(tweet_text, '"')
                    self.tweets[person][tweet]['urlc'] = self.calculate_word_occ(tokens, 'url')
                    self.tweets[person][tweet]['captc'] = self.calculate_number_uppercase_words(tweet_text)

    def get_results(self):
        dates = {}
        dates['R'], dates['D'], dates['L'] = [], [], []
        for party in self.processed_tweets.keys():
            party_ctr = 0
            amount_results = 0
            compound, pos, neu, neg = 0, 0, 0, 0
            charc, wordc = 0, 0
            excc, quesc, quotec, urlc, captc = 0, 0, 0, 0, 0
            word_grams = {}
            word_grams['1_gram'], word_grams['2_gram'], word_grams['3_gram'], word_grams['4_gram'] = [], [], [], []
            for item in self.processed_tweets[party]:
                for element in self.processed_tweets[party][item]:
                    party_ctr += 1
                    amount_results += 1
                    compound += self.processed_tweets[party][item][element]['compound']
                    pos += self.processed_tweets[party][item][element]['pos']
                    neg += self.processed_tweets[party][item][element]['neg']
                    neu += self.processed_tweets[party][item][element]['neu']
                    charc += self.processed_tweets[party][item][element]['charc']
                    wordc += self.processed_tweets[party][item][element]['wordc']
                    excc += self.processed_tweets[party][item][element]['excc']
                    quesc += self.processed_tweets[party][item][element]['quesc']
                    quotec += self.processed_tweets[party][item][element]['quotec']
                    urlc += self.processed_tweets[party][item][element]['urlc']
                    captc += self.processed_tweets[party][item][element]['captc']
                    word_grams['1_gram'].extend(self.processed_tweets[party][item][element]['1_gram'])
                    word_grams['2_gram'].extend(self.processed_tweets[party][item][element]['2_gram'])
                    word_grams['3_gram'].extend(self.processed_tweets[party][item][element]['3_gram'])
                    word_grams['4_gram'].extend(self.processed_tweets[party][item][element]['4_gram'])
                    dates[party].append(self.processed_tweets[party][item][element]['date'])
            print('Class: ' + party + ' contains ' + str(party_ctr) + ' tweets')
            print('Average compound: ' + party + ' ' + str(compound/amount_results))
            print('Average pos: ' + party + ' ' + str(pos / amount_results))
            print('Average neg: ' + party + ' ' + str(neg / amount_results))
            print('Average neu: ' + party + ' ' + str(neu / amount_results))
            print('Average charc: ' + party + ' ' + str(charc / amount_results))
            print('Average wordc: ' + party + ' ' + str(wordc / amount_results))
            print('Average excc: ' + party + ' ' + str(excc / amount_results))
            print('Average quesc: ' + party + ' ' + str(quesc / amount_results))
            print('Average quotec: ' + party + ' ' + str(quotec / amount_results))
            print('Average urlc: ' + party + ' ' + str(urlc / amount_results))
            print('Average captc: ' + party + ' ' + str(captc / amount_results))
            counts1, counts2, counts3, counts4 = Counter(word_grams['1_gram']), Counter(word_grams['2_gram']), \
                                                 Counter(word_grams['3_gram']), Counter(word_grams['4_gram'])
            print(counts1)
            print(counts2)
            print(counts3)
            print(counts4)
        print(dates)

    @staticmethod
    def calculate_word_grams(tokens, n=1):
        s = []
        for ng in ngrams(tokens, n):
            s.append(' '.join(str(i) for i in ng))
        return s

    @staticmethod
    def calculate_sentiment(tokens, sid=default_sid):
        sentence = ' '.join(token for token in tokens)
        return sid.polarity_scores(sentence)

    @staticmethod
    def calculate_character_count(tweet):
        return len(tweet)

    @staticmethod
    def calculate_word_count(tokens):
        return len(tokens)

    @staticmethod
    def calculate_symbol_occ(tweet, symbol):
        return tweet.count(symbol)

    @staticmethod
    def calculate_word_occ(tokens, word):
        return tokens.count(word)

    def calculate_number_uppercase_words(self, tweet):
        counter = 0
        for word in tweet.split():
            if word.isupper():
                contains = False
                for item in self.exception_list:
                    if item in word:
                        contains = True
                if not contains:
                    counter += 1
        return counter


a = Analyzer()
a.calc_features()
a.save()
a.get_results()