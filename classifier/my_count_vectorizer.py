from tokenizer.regex import Tokenizer
import numpy as np


class MyCountVectorizer:

    def __init__(self):
        self.save = {}
        self.index_unk_word = 0
        self.cnt = 1
        self.list_unk_word = []
        return

    def fit_transform(self, list_string):

        # calculate type of token
        tokenizer = Tokenizer()
        counter_token = {}

        for string in list_string:

            list_token = tokenizer.start_tokenize(string)

            for token in list_token:
                if self.save.get(token["token"]) is None:
                    counter_token[token["token"]] = 0
                    self.save[token["token"]] = self.cnt
                    self.cnt += 1
                else:
                    counter_token[token["token"]] += 1

        self.list_unk_word = sorted(counter_token)

        # unk word is the top 10 words
        self.list_unk_word = self.list_unk_word[:10]

        data_transform = np.zeros(shape=(len(list_string), self.cnt))

        for i in range(len(list_string)):

            list_token = tokenizer.start_tokenize(string)
            for token in list_token:

                if self.save.get(token["token"]) not in self.list_unk_word:
                    data_transform[i][self.save.get(token["token"])] += 1
                else:
                    data_transform[i][self.index_unk_word] += 1

        return data_transform

    def transform_individual_sentence(self, string):

        data_transform = np.zeros(shape=self.cnt)

        tokenizer = Tokenizer()
        list_token = tokenizer.start_tokenize(string)

        for token in list_token:
            ind = self.save.get(token["token"])

            if ind is None or self.save.get(token["token"]) in self.list_unk_word:
                ind = self.index_unk_word

            data_transform[ind] += 1

        return data_transform
