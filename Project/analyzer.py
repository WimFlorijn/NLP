from tokenizer import Tokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.util import ngrams
import os, glob, json, shutil

default_sid = SentimentIntensityAnalyzer()

DEFAULT_DATA_DIR = "data"
DEFAULT_OUTPUT_DATA_DIR = "processed_data"


class Analyzer:
    def __init__(
            self,
            data_dir=DEFAULT_DATA_DIR,
            output_dir=DEFAULT_OUTPUT_DATA_DIR):
        """
        Initializes the analyzer class with a data set.

        :param data_dir: the directory to which tweets are saved.
        """

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.tweets = {}
        self.pruned_tweets = {}
        self.inverse_index = {}

        # Load the previously downloaded tweets
        self.load()
        self.prune()

    def load(self):

        # Do not attempt to load tweets if the data directory does not exist
        if not os.path.exists(self.data_dir):
            return

        # Load the tweets into memory
        self.tweets = {}
        for file_name in glob.glob(os.path.join(self.data_dir, '*.json')):
            with open(file_name) as f:
                self.tweets[os.path.basename(file_name[:-5])] = json.load(f)

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
            for item in self.tweets[person]:
                tweet = item['text']
                self.add_to_inverse_index(regex_tokenizer.tokenize(tweet))
                self.pruned_tweets[person][tweet] = {}

    def calc_features(
            self,
            sentiment_analysis_feature=True,
            n_gram_feature=True,
            count_feature=True,
            punctuation_feature=True):

        for person in self.pruned_tweets:
            for tweet in self.pruned_tweets[person]:
                tokens = regex_tokenizer.tokenize(tweet)
                if sentiment_analysis_feature:
                    self.pruned_tweets[person][tweet]['sentiment_analysis_weight'] = Analyzer.calculate_sentiment(tokens)

                if n_gram_feature:
                    score = 0
                    for token in tokens:
                        score += 1/self.inverse_index[token]
                    self.pruned_tweets[person][tweet]['1_gram'] = score
                    for i in range(1,5):
                        score = 0
                        if i > 1:
                            n_grams = Analyzer.calculate_word_grams(tokens, n=i)
                            n_gram_tokens = list()
                            for value in n_grams:
                                n_gram_tokens.extend(value.split())
                        else:
                            n_gram_tokens = tokens
                        for token in n_gram_tokens:
                            amount = (1 / self.inverse_index[token])
                            score += amount
                        self.pruned_tweets[person][tweet][str(i)+'_gram'] = score


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

    @staticmethod
    def calculate_number_capitalized_words(tweet):
        ctr = 0
        for word in tweet.split():
            if word.isupper():
                ctr += 1
        return ctr


regex_tokenizer = Tokenizer()
analyzer = Analyzer()

analyzer.calc_features()
analyzer.save()

test_tweet = "VADER is very smart, handsome, and funny."
tweet_tokens = regex_tokenizer.tokenize(test_tweet)

print(Analyzer.calculate_sentiment(tweet_tokens))

print(Analyzer.calculate_word_grams(tweet_tokens, n=3))

print(Analyzer.calculate_character_count(test_tweet))

print(Analyzer.calculate_word_count(tweet_tokens))

print(Analyzer.calculate_symbol_count(test_tweet, '.'))

print(Analyzer.calculate_number_capitalized_words(test_tweet))
