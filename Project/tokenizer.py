from nltk.tokenize import RegexpTokenizer


class Tokenizer:
    def __init__(self):
        self.tokenizer = RegexpTokenizer(r'\w+')

    def tokenize(self, text, custom_tokenizer=None, cast_to_lower=True):
        if custom_tokenizer is None:
            result = self.tokenizer.tokenize(text)
        else:
            result = custom_tokenizer.tokenize(text)
        if cast_to_lower:
            result = [r.lower() for r in result]
        return result
