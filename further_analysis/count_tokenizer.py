"""
Count and sort result of the new tokenizer
"""

import json
import operator
import pycrfsuite

from tokenizer.regex import Tokenizer
from tokenizer import crf

def count_token_regex(data_dir):
    """ get token count using new regex tokenizer """

    re_tokenizer = Tokenizer()
    re_tokenizer.start_tokenize_from_folder(data_dir)

    with open(data_dir + 'list_string_regex_token.json') as data_file:
        data = json.load(data_file)

    with open(data_dir + "stop_words.json") as stop_files:
        stop_words = json.load(stop_files)

    word_count = {}

    for item in data:
        for element in item:
            token = element['origin']
            tag = element['token']

            if tag.upper() != "JAVA":
                continue

            if token.lower() in stop_words:
                continue

            if token in word_count:
                word_count[token] += 1
            else:
                word_count[token] = 1

    word_count = sorted(word_count.items(), key=operator.itemgetter(1), reverse=True)

    with open(data_dir + 'result_new_token_regex_count.json', 'w') as result:
        json.dump(word_count, result, indent=4)

def count_token_crf(data_dir):
    """ get token count using new crf tokenizer """

    model_name = crf.train_tokenizer(data_dir, data_dir + "crf_model")
    tagger = pycrfsuite.Tagger()
    tagger.open(model_name)

    with open(data_dir + 'data.json') as data_file:
        data = json.load(data_file)

    with open(data_dir + "stop_words.json") as stop_files:
        stop_words = json.load(stop_files)

    word_count = {}

    for item in data["list_string"]:
        item_token = crf.tokenize_from_model(tagger, item)
        for token in item_token:
            if token.lower() in stop_words:
                continue
            if token in word_count:
                word_count[token] += 1
            else:
                word_count[token] = 1

    word_count = sorted(word_count.items(), key=operator.itemgetter(1), reverse=True)

    with open(data_dir + 'result_new_token_crf_count.json', 'w') as result:
        json.dump(word_count, result, indent=4)

if __name__ == "__main__":
    count_token_regex("../data/")
    count_token_crf("../data/")
