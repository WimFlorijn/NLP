import os
import json
import shutil

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.util import ngrams

from dataset import TwitterDataSet
from tokenizer import Tokenizer

default_sid = SentimentIntensityAnalyzer()

DEFAULT_CONFIG_DIR = "config"
DEFAULT_DATA_DIR = "data"
DEFAULT_OUTPUT_DATA_DIR = "processed_data"
DEFAULT_EXCEPTION_LIST = ['HASH_TAG', 'URL', 'I']


class Analyzer:

    tokenizer = Tokenizer()

    def __init__(
            self,
            config_dir=DEFAULT_CONFIG_DIR,
            data_dir=DEFAULT_DATA_DIR,
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
        self.pruned_tweets = {}
        self.inverse_index = {}

        # Load the previously downloaded tweets
        self.load()
        self.prune()

    def load(self):

        data_set = TwitterDataSet(self.config_dir, self.data_dir)
        self.tweets = data_set.get_preprocessed_tweets()

    def save(self):
        """
        Saves the currently loaded tweets to the data directory.
        """

        # Create the data directory if it exists, then create it anew
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.mkdir(self.output_dir)

        # Save every tweet by author
        for author, tweets in self.pruned_tweets.items():
            with open(os.path.join(self.output_dir, author + '.json'), 'w') as f:
                json.dump(tweets, f)

    def add_to_inverse_index(self, tokens):
        for token in tokens:
            if token in self.inverse_index:
                self.inverse_index[token] += 1
            else:
                self.inverse_index[token] = 1

    def prune(self):
        for person in self.tweets:
            self.pruned_tweets[person] = {}
            for tweet_id, item in self.tweets[person].items():
                tweet = item['text']
                self.add_to_inverse_index(self.tokenizer.tokenize(tweet))
                self.pruned_tweets[person][tweet] = {}

    def calc_features(
            self,
            sentiment_analysis_feature=True,
            n_gram_feature=True,
            count_feature=True,
            punctuation_feature=True):

        for person in self.pruned_tweets:
            for tweet in self.pruned_tweets[person]:
                tokens = self.tokenizer.tokenize(tweet)
                if sentiment_analysis_feature:
                    ss = self.calculate_sentiment(tokens)
                    for k in sorted(ss):
                        self.pruned_tweets[person][tweet][k] = ss[k]

                if n_gram_feature:
                    for i in range(1,5):
                        score = 0
                        if i > 1:
                            n_grams = self.calculate_word_grams(tokens, n=i)
                            n_gram_tokens = list()
                            for value in n_grams:
                                n_gram_tokens.extend(value.split())
                        else:
                            n_gram_tokens = tokens
                        for token in n_gram_tokens:
                            score += 1 / self.inverse_index[token]
                        self.pruned_tweets[person][tweet][str(i)+'_gram'] = score

                if count_feature:
                    self.pruned_tweets[person][tweet]['charc'] = self.calculate_character_count(tweet)
                    self.pruned_tweets[person][tweet]['wordc'] = self.calculate_word_count(tokens)

                if punctuation_feature:
                    self.pruned_tweets[person][tweet]['excc'] = self.calculate_symbol_count(tweet, '!')
                    self.pruned_tweets[person][tweet]['quesc'] = self.calculate_symbol_count(tweet, '?')
                    self.pruned_tweets[person][tweet]['quotec'] = self.calculate_symbol_count(tweet, '"')
                    self.pruned_tweets[person][tweet]['captc'] = self.calculate_number_uppercase_words(tweet)

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
    def calculate_symbol_count(tweet, symbol):
        return tweet.count(symbol)

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
