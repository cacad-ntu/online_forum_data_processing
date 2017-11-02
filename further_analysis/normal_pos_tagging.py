import nltk
import json
from tokenizer.regex import Tokenizer


def pos_tag(data_dir):
    my_arr = []

    with open(data_dir + "data.json") as test:
        data = json.load(test)

    for sentence in data['list_string']:
        sentence = sentence.strip()
        tokenizer = Tokenizer()
        sentence_token = tokenizer.start_tokenize(sentence, just_return_list=True)
        tag = nltk.pos_tag(sentence_token)
        my_dict = {}
        my_dict["origin"] = sentence
        my_dict["pos_tag"] = tag
        my_arr.append(my_dict)

    with open(data_dir + "pos_tag_normal.json", "w") as out_file:
        json.dump(my_arr, out_file)

if __name__ == "__main__":

    pos_tag("../data/")
