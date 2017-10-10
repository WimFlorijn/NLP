from tokenizer import Tokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.util import ngrams

default_sid = SentimentIntensityAnalyzer()


def calculate_word_grams(sentence, n=1):
    s = []
    for ng in ngrams(sentence, n):
        s.append(' '.join(str(i) for i in ng))
    return s


def calculate_sentiment(sentence, sid=default_sid):
    return sid.polarity_scores(sentence)


def calculate_character_count(sentence):
    return len(sentence)


def calculate_word_count(sentence):
    return len(sentence.split())


def calculate_symbol_count(sentence, symbol):
    return sentence.count(symbol)


def calculate_number_capitalized_words(sentence):
    ctr = 0
    for word in sentence.split():
        if word.isupper():
            ctr += 1
    return ctr


if __name__ == "__main__":
    regex_tokenizer = Tokenizer()
    example = "VADER is very smart, handsome, and funny."
    ss = calculate_sentiment(example)

    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end='')

    print(calculate_word_grams('one two three four'.split(' '), n=3))

    print(calculate_character_count(example))

    print(calculate_word_count(example))

    print(calculate_symbol_count(example, '.'))

    print(calculate_number_capitalized_words(example))
