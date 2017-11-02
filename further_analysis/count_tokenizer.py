"""
Count and sort result of the new tokenizer
"""

import json
import operator

from tokenizer.regex import Tokenizer

def count_token(data_dir):
    """ get token count using new tokenizer """

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

    with open(data_dir + 'result_new_token_count.json', 'w') as result:
        json.dump(word_count, result, indent=4)

if __name__ == "__main__":
    count_token("../data/")
