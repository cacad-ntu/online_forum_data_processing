from tokenizer.regex import Tokenizer
import numpy as np


class MyCountVectorizer:

    def __init__(self):
        return

    def fit_transform(self, list_string):

        # calculate type of token
        tokenizer = Tokenizer()
        cnt = 0

        save = {}

        for string in list_string:

            list_token = tokenizer.start_tokenize(string)

            for token in list_token:
                if save.get(token["token"]) is None:
                    save[token["token"]] = cnt
                    cnt += 1

        data_transform = np.zeros(shape=(len(list_string), cnt))

        for i in range(len(list_string)):

            list_token = tokenizer.start_tokenize(string)
            for token in list_token:

                data_transform[i][save.get(token["token"])] += 1

        return data_transform
