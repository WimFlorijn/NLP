import glob
import json
import os
import re
import shutil

from nltk.twitter import credsfromfile, Query

DEFAULT_CONFIG_DIR = "config"
DEFAULT_DATA_DIR = "data"

CREDS_FILE = "credentials.txt"
SELECTED_TWEETS_FILE = "selected_tweets.txt"


class TwitterDataSet:

    def __init__(
            self,
            config_dir=DEFAULT_CONFIG_DIR,
            data_dir=DEFAULT_DATA_DIR):
        """
        Initializes the data set with what data has already been downloaded.

        :param config_dir: directory containing the Twitter configuration.
        :param data_dir: the directory to which tweets are saved.
        """

        self.oauth = credsfromfile(CREDS_FILE, config_dir)
        self.config_dir = config_dir
        self.data_dir = data_dir
        self.users = {}
        self.tweets = {}

        # Load the previously downloaded tweets
        self.load()

    def add_tweets(self, user, party):
        """
        Downloads tweets from a single Twitter user up to the specified ID.

        :param user: the Twitter handle of the user.
        :param party: the political party to which `user` belongs.
        :return the list of downloaded tweets.
        """

        query = Query(**self.oauth)
        tweets = query.get_user_timeline(
            screen_name=user,
            count=200,
            exclude_replies='false',
            include_rts='true')
        self.tweets[user] = tweets
        self.save()
        self.users[user] = party
        return tweets

    def save(self):
        """
        Saves the currently loaded tweets to the data directory.
        """

        # Create the data directory if it exists, then create it anew
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)
        os.mkdir(self.data_dir)

        # Save every tweet by author
        for author, tweets in self.tweets.items():
            with open(os.path.join(self.data_dir, author + '.json'), 'w') as f:
                json.dump(tweets, f)

    def load(self):

        # Do not attempt to load tweets if the data directory does not exist
        if not os.path.exists(self.data_dir):
            return

        # Load the tweets into memory
        self.tweets = {}
        for file_name in glob.glob(os.path.join(self.data_dir, '*.json')):
            with open(file_name) as f:
                self.tweets[os.path.basename(file_name[:-5])] = json.load(f)

        # Load the users' political parties
        with open(os.path.join(self.config_dir, 'users.txt')) as users:
            for line in users.readlines():
                if line.lstrip().startswith('#') or len(
                        line.strip()) == 0:
                    continue
                party, user = line.strip().split(':', 1)
                self.users[user] = party

    def get_preprocessed_tweets(self):

        # Load the list of tweets that were selected
        selected_tweet_ids = set()
        try:
            with open(os.path.join(self.config_dir, SELECTED_TWEETS_FILE)) as f:
                for line in f.readlines():
                    if line.lstrip().startswith('@') or len(line.strip()) == 0:
                        continue
                    try:
                        tweet_id = int(line.strip())
                    except ValueError:
                        print(f"[ERROR] Invalid tweet ID: '{line.strip()}'")
                        continue
                    selected_tweet_ids.add(tweet_id)
        except IOError:
            print(
                "[WARNING] selected_tweets.txt not found, tweets will not be "
                "filtered."
            )
            selected_tweet_ids = None

        # Preprocess the tweets
        preprocessed_tweets = {}
        for author, tweets in self.tweets.items():
            preprocessed_tweets[author] = {
            }
            for tweet in tweets:
                if selected_tweet_ids is not None \
                        and tweet['id'] not in selected_tweet_ids:
                    continue

                # Preprocess the tweet's text
                text = tweet['text'].lower()
                text = re.sub(r'#[\w\d]+', 'HASH_TAG', text)
                for url in reversed(tweet['entities']['urls']):
                    text = text[:url['indices'][0]] + 'URL' + text[url['indices'][1]:]

                preprocessed_tweets[author][tweet['id']] = {
                    'date': tweet['created_at'],
                    'hashtags': tuple(h['text'] for h in tweet['entities']['hashtags']),
                    'urls': tuple(u['expanded_url'] for u in tweet['entities']['urls']),
                    'party': self.users[author],
                    'text': text
                }

        return preprocessed_tweets
