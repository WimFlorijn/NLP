import json

from Project.dataset import TwitterDataSet

if __name__ == '__main__':
    # Load the downloaded data set
    data_set = TwitterDataSet()
    processed_data_set = data_set.get_preprocessed_tweets()
    with open('preprocessed_data_set.json', 'w') as f:
        json.dump(processed_data_set, f, indent=4)

    # TODO Analyze these tweets
    democrats = republicans = libertarians = 0
    for user, tweets in processed_data_set.items():
        print(f'@{user}: {len(tweets)} tweets')
        for tweet_id, tweet in tweets.items():
            if tweet['party'] == 'D':
                democrats += 1
            elif tweet['party'] == 'R':
                republicans += 1
            elif tweet['party'] == 'L':
                libertarians += 1
            print(f"\t[{tweet['date'][:-11]}] {tweet['text']}")
    print(
        f"Democrats: {democrats}\n"
        f"Republicans: {republicans}\n"
        f"Libertarians: {libertarians}"
    )
