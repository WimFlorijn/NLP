import os

from .twitter import TwitterDataSet


def build_data_set():
    """
    Builds a data set from Twitter.
    """

    data_set = TwitterDataSet()
    users_file = os.path.join(data_set.config_dir, 'users.txt')

    # Read the list of users from the users.txt file
    with open(users_file) as users:
        for line in users.readlines():
            if line.lstrip().startswith('#') or len(line.strip()) == 0:
                continue
            party, user = line.strip().split(':', 1)
            print(f'Downloading tweets from {user} ({party})...')
            tweets = data_set.add_tweets(user, party)
            print(f'Downloaded {len(tweets)} tweets from {user} ({party}).')

    return data_set
