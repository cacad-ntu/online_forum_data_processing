"""
Count and sort result of the new tokenizer
"""

import json
import operator
import pycrfsuite

from tokenizer.regex import Tokenizer
from tokenizer import crf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np


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

    with open(data_dir + 'train_data.json') as train_file:
        train_data = json.load(train_file)

    cnt = 0
    accuracy_ave = []
    f1_ave = []
    precision_ave = []
    recall_ave = []

    for item in data["list_string"]:

        data_compare = []
        if cnt <= 99:
            for token in train_data[cnt]['pos_tag']:
                if len(token) > 0:
                    data_compare.append(token[0])

        cnt += 1

        item_token = crf.tokenize_from_model(tagger, item)

        if cnt <= 99:
            while len(item_token) != len(data_compare):
                if len(item_token) < len(data_compare):
                    item_token.append('')
                else:
                    data_compare.append('')

        if cnt <= 99:
            accuracy_ave.append(accuracy_score(data_compare, item_token))
            f1_ave.append(f1_score(data_compare, item_token, average='macro'))
            precision_ave.append(precision_score(data_compare, item_token, average='macro'))
            recall_ave.append(recall_score(data_compare, item_token, average='macro'))

        for token in item_token:
            if token.lower() in stop_words:
                continue
            if token in word_count:
                word_count[token] += 1
            else:
                word_count[token] = 1

    word_count = sorted(word_count.items(), key=operator.itemgetter(1), reverse=True)
    print "accuracy: %s \n" % np.mean(accuracy_ave)
    print "f1: %s \n" % np.mean(f1_ave)
    print "precision: %s \n" % np.mean(precision_ave)
    print "recall: %s \n" % np.mean(recall_ave)

    with open(data_dir + 'result_new_token_crf_count.json', 'w') as result:
        json.dump(word_count, result, indent=4)

if __name__ == "__main__":
    count_token_regex("../data/")
    count_token_crf("../data/")
