""" CRF TOKENIZER """

import json
import sys
import pycrfsuite


def char_features(sentence, index, window_size=4):
    """ Extract features from words """

    char = sentence[index]
    features = [
        'Char = ' + char.lower(),
        'Char is in uppercase   = %s' % char.isupper(),
        'Char is a number       = %s' % char.isdigit()
    ]

    window_left = max(0, index-window_size)
    if index > 0:
        for i in range(window_left, index):
            features.extend([
                'Previous ' + str(index-i) + ' char = ' + sentence[i].lower(),
                'Previous ' + str(index-i) + ' char is a number= %s' % sentence[i].isdigit(),
                'Previous ' + str(index-i) + ' char is in uppercase  = %s' % sentence[i].isupper()
            ])
    else:
        features.append('Start')

    window_right = min(len(sentence)-1, index+window_size)
    if index > len(sentence)-1:
        for i in range(index+1, window_right+1):
            features.extend([
                'Next ' + str(window_right-i) + ' char = ' + sentence[i].lower(),
                'Next ' + str(window_right-i) + ' char is a number= %s' % sentence[i].isdigit(),
                'Next ' + str(window_right-i) + ' char is in uppercase  = %s' % sentence[i].isupper()
            ])
    else:
        features.append('End')

    return features

def sentence_to_chars(sentence):
    """ Extract character feature from sentences """
    return [char_features(sentence, index) for index in range(len(sentence))]

def sentence_tag(tag_list):
    """ Extract IOB tag from tag sentences """
    return [iob_tag for iob_tag in tag_list]

def create_IOB_tag(original_string, pos_tag):
    """ generate IOB tag from pos tag (for train tokenizer)"""
    BEGINNING_TAG = "S"
    IN_TAG = "I"
    OUT_TAG = "O"

    iob_tag = ""
    cur = 0
    for token_tag in pos_tag:
        if token_tag[0] == "''" or token_tag[0] == "``":
            token_tag[0] = '"'
        while original_string[cur] != token_tag[0][0]:
            if token_tag[0] == '"' and original_string[cur:cur+2] == "''":
                token_tag[0] = "''"
                break
            if token_tag[0] == '"' and original_string[cur:cur+2] == "``":
                token_tag[0] = "``"
                break
            cur += 1
            iob_tag += OUT_TAG
        cur += len(token_tag[0])
        iob_tag += BEGINNING_TAG + IN_TAG*(len(token_tag[0]) - 1)
        if token_tag[1] == ".":
            BEGINNING_TAG = "S"
        else:
            BEGINNING_TAG = "T"

    return iob_tag

def train_tokenizer(data_dir, model_name, k_fold=5):
    """ train data for tokenizer """
    with open(data_dir+"train_data.json") as json_data:
        data = json.load(json_data)

    # Generate IOB tag from POS tag
    iob_tag = []
    for item in data:
        iob_tag.append(create_IOB_tag(item["origin"], item["pos_tag"]))

    x_train = [sentence_to_chars(item["origin"]) for item in data]
    y_train = [sentence_tag(tag) for tag in iob_tag]

    trainer = pycrfsuite.Trainer()

    for xseq, yseq in zip(x_train, y_train):
        trainer.append(xseq, yseq)

    trainer.set_params({
        'c1': 1.0,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 100,  # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })

    trainer.train(model_name+".crfsuite")
    return model_name+".crfsuite"

def tokenize_from_model(tagger, sentence):
    """ tokenize data set from folder """
    list_of_token = []
    iob_tag = tagger.tag(sentence_to_chars(sentence))
    for cur in range(len(sentence)):
        start = cur
        if iob_tag[start] == "S" or iob_tag[start] == "T":
            end = start+1
            if end >= len(sentence):
                list_of_token.append(sentence[start])
                break
            while iob_tag[end] == "I":
                end += 1
            list_of_token.append(sentence[start:end])
            cur = end-1
    return list_of_token

if __name__ == '__main__':
    train_tokenizer("../data/", "../data/crf_model")
