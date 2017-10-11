from Project.dataset import TwitterDataSet

if __name__ == '__main__':
    # Load the downloaded data set
    data_set = TwitterDataSet()
    processed_data_set = data_set.get_preprocessed_tweets()

    # TODO Analyze these tweets
    for user, tweets in processed_data_set.items():
        print(f'@{user}: {len(tweets)} tweets')
        for tweet_id, tweet in tweets.items():
            print(f"\t[{tweet['date'][:-11]}] {tweet['text']}")
