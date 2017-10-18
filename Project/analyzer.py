import os
import json
import shutil

from collections import Counter
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.util import ngrams

from .dataset import TwitterDataSet
from .tokenizer import Tokenizer

default_sid = SentimentIntensityAnalyzer()

DEFAULT_CONFIG_DIR = "Project/config"
DEFAULT_INPUT_DATA_DIR = "data"
DEFAULT_OUTPUT_DATA_DIR = "processed_data"
DEFAULT_RESULT_DIR = "results"
DEFAULT_EXCEPTION_LIST = ['HASH_TAG', 'URL', 'I', 'AMP']
MAX_N_GRAM = 4
TOP_N_VALUES = 5


class Analyzer:

    tokenizer = Tokenizer()

    def __init__(
            self,
            config_dir=DEFAULT_CONFIG_DIR,
            data_dir=DEFAULT_INPUT_DATA_DIR,
            output_dir=DEFAULT_OUTPUT_DATA_DIR,
            result_dir=DEFAULT_RESULT_DIR):
        """
        Initializes the analyzer class with a configuration and a data set.

        :param config_dir: directory containing the Twitter configuration.
        :param data_dir: the directory to which tweets are saved.
        :param output_dir: the directory to which the processed tweets and results are saved.
        """

        self.config_dir = config_dir
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.result_dir=result_dir
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

            for party in elements.keys():
                with open(os.path.join(self.output_dir, party + '.json'), 'w') as f:
                    json.dump(elements[party], f)

    def calc_features(
            self,
            sentiment_analysis_feature=True,
            n_gram_feature=True,
            count_feature=True,
            punctuation_feature=True):
        """
        Calculates the different feature types delivered as @params.

        :param sentiment_analysis_feature: Indicates whether sentiment analysis should be performed.
        :param n_gram_feature: Indicates whether word/n -grams should be calculated.
        :param count_feature: Indicates whether tweet character/word counts should be performed.
        :param punctuation_feature: Indicates whether the punctuation feature should be calculated.
        """

        for person in self.tweets:
            for tweet in self.tweets[person]:
                tweet_text = self.tweets[person][tweet]['text']
                tokens = self.tokenizer.tokenize(tweet_text)
                if sentiment_analysis_feature:
                    ss = self.calculate_sentiment(tokens)
                    for k in sorted(ss):
                        self.tweets[person][tweet][k] = ss[k]

                if n_gram_feature:
                    for i in range(1, MAX_N_GRAM + 1):
                        if i == 1:
                            exception_list_lowercase = [item.lower() for item in DEFAULT_EXCEPTION_LIST]
                            n_gram_tokens = [word for word in tokens if word not in stopwords.words('english')
                                               and word not in exception_list_lowercase]
                        else:
                            n_gram_tokens = self.calculate_word_grams(tokens, n=i)
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

    def get_results(
            self,
            sentiment_analysis_feature=True,
            n_gram_feature=True,
            count_feature=True,
            punctuation_feature=True):
        """
        Calculates the results based on the feature factors.

        :param sentiment_analysis_feature: Indicates whether sentiment analysis results should be calculated.
        :param n_gram_feature: Indicates whether word/n -grams results should be calculated.
        :param count_feature: Indicates whether tweet character/word counts results should be calculated.
        :param punctuation_feature: Indicates whether the punctuation feature results should be calculated.
        """

        # Create the data directory if it exists, then create a new
        if os.path.exists(self.result_dir):
            shutil.rmtree(self.result_dir)
        os.mkdir(self.result_dir)

        results = dict()
        for party in self.processed_tweets.keys():
            results[party] = dict()
            results[party]['creation_dates'], results[party]['hashtags'] = [], []
            results[party]['tweet_count'] = \
                sum(len(self.processed_tweets[party][item]) for item in self.processed_tweets[party])

            if sentiment_analysis_feature:
                results[party]['avg_compound'], results[party]['avg_pos'], \
                    results[party]['avg_neu'], results[party]['avg_neg'] = 0, 0, 0, 0

            if n_gram_feature:
                results[party]['1_grams'], results[party]['2_grams'], results[party]['3_grams'], \
                results[party]['4_grams'] = [], [], [], []

            if count_feature:
                results[party]['avg_charc'], results[party]['avg_wordc'] = 0, 0

            if punctuation_feature:
                results[party]['avg_excc'], results[party]['avg_quesc'], results[party]['avg_quotec'], \
                    results[party]['avg_urlc'], results[party]['avg_captc'] = 0, 0, 0, 0, 0

            for item in self.processed_tweets[party]:
                for element in self.processed_tweets[party][item]:
                    results[party]['creation_dates'].append(self.processed_tweets[party][item][element]['date'])
                    results[party]['hashtags'].extend(self.processed_tweets[party][item][element]['hashtags'])

                    if sentiment_analysis_feature:
                        results[party]['avg_compound'] += \
                            self.processed_tweets[party][item][element]['compound'] / results[party]['tweet_count']
                        results[party]['avg_pos'] += \
                            self.processed_tweets[party][item][element]['pos'] / results[party]['tweet_count']
                        results[party]['avg_neu'] += \
                            self.processed_tweets[party][item][element]['neu'] / results[party]['tweet_count']
                        results[party]['avg_neg'] += \
                            self.processed_tweets[party][item][element]['neg'] / results[party]['tweet_count']

                    if n_gram_feature:
                        results[party]['1_grams'].extend(self.processed_tweets[party][item][element]['1_gram'])
                        results[party]['2_grams'].extend(self.processed_tweets[party][item][element]['2_gram'])
                        results[party]['3_grams'].extend(self.processed_tweets[party][item][element]['3_gram'])
                        results[party]['4_grams'].extend(self.processed_tweets[party][item][element]['4_gram'])

                    if count_feature:
                        results[party]['avg_charc'] += \
                            self.processed_tweets[party][item][element]['charc'] / results[party]['tweet_count']
                        results[party]['avg_wordc'] += \
                            self.processed_tweets[party][item][element]['wordc'] / results[party]['tweet_count']

                    if punctuation_feature:
                        results[party]['avg_excc'] += \
                            self.processed_tweets[party][item][element]['excc'] / results[party]['tweet_count']
                        results[party]['avg_quesc'] += \
                            self.processed_tweets[party][item][element]['quesc'] / results[party]['tweet_count']
                        results[party]['avg_quotec'] += \
                            self.processed_tweets[party][item][element]['quotec'] / results[party]['tweet_count']
                        results[party]['avg_urlc'] += \
                            self.processed_tweets[party][item][element]['urlc'] / results[party]['tweet_count']
                        results[party]['avg_captc'] += \
                            self.processed_tweets[party][item][element]['captc'] / results[party]['tweet_count']

            results[party]['hashtags'] = Counter(results[party]['hashtags']).most_common(TOP_N_VALUES)

            if n_gram_feature:
                results[party]['1_grams'] = Counter(results[party]['1_grams']).most_common(TOP_N_VALUES)
                results[party]['2_grams'] = Counter(results[party]['2_grams']).most_common(TOP_N_VALUES)
                results[party]['3_grams'] = Counter(results[party]['3_grams']).most_common(TOP_N_VALUES)
                results[party]['4_grams'] = Counter(results[party]['4_grams']).most_common(TOP_N_VALUES)

        with open(os.path.join(self.result_dir, 'results.json'), 'w') as f:
            json.dump(results, f)

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


#a = Analyzer()
#a.calc_features()
#a.save()
#a.get_results()
