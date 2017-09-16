import nltk
import os

class Tokenizer:
    def __init__(self, location):
        self.location = location

    @staticmethod
    def tokenize(text):
        tokens = nltk.word_tokenize(text)
        return tokens

    @staticmethod
    def tokenizeFolder(location):
        for file in os.listdir(location):
            filename = os.fsdecode(file)
            if filename.endswith(".txt"):
                with open(location + filename, 'r', encoding='utf-8') as file:
                    content = file.read()
                    print(Tokenizer.tokenize(content))

if __name__ == "__main__":

    #Question 4
    text1 = 'I am so blue I\'m greener than purple.'
    text2 = 'I stepped on a Corn Flake, now I\'m a Cereal Killer'
    text3 = 'I like the course Natural Language Processing'

    print(Tokenizer.tokenize(text1))
    print(Tokenizer.tokenize(text2))
    print(Tokenizer.tokenize(text3))


    #datadir = 'dataset/blogs/'
    #train =  'train/'
    #test = 'test/'

    #location = datadir + train
    #tokenizer = Tokenizer(location)
    #Tokenizer.tokenizeFolder(location)


