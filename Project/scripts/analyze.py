from Project.analyzer import Analyzer
from Project.tokenizer import Tokenizer

if __name__ == '__main__':
    regex_tokenizer = Tokenizer()
    analyzer = Analyzer()

    analyzer.calc_features()
    analyzer.save()

    test_tweet = "VADER is very smart, handsome, and funny."
    tweet_tokens = regex_tokenizer.tokenize(test_tweet)

    print(f"Sentiment: {Analyzer.calculate_sentiment(tweet_tokens)}")

    print(f"N-grams (3): {Analyzer.calculate_word_grams(tweet_tokens, n=3)}")

    print(f"Character count: {Analyzer.calculate_character_count(test_tweet)}")

    print(f"Word count: {Analyzer.calculate_word_count(tweet_tokens)}")

    print(f"Punctuation (.): {Analyzer.calculate_symbol_count(test_tweet, '.')}")

    print(f"Uppercase words: {analyzer.calculate_number_uppercase_words(test_tweet)}")
