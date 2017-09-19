from nltk.tokenize import RegexpTokenizer

import os


class Tokenizer:
    def __init__(self):
        self.tokenizer = RegexpTokenizer(r'\w+')

    def tokenize(self, text, custom_tokenizer=None):
        if custom_tokenizer is None:
            result = self.tokenizer.tokenize(text)
        else:
            result = custom_tokenizer.tokenize(text)
        return result

    def tokenize_folder(self, location, gender='All'):
        token_text = []
        for text_file in os.listdir(location):
            text_file_name = os.fsdecode(text_file)
            if gender == 'All' or text_file_name.startswith(gender):
                with open(location + text_file_name, 'r', encoding='utf-8') as usable_text_file:
                    file_content = usable_text_file.read()
                    token_text.extend(self.tokenize(file_content))
        return token_text






