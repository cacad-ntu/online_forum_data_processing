""" CRF POS TAGGER """

import sklearn
import pycrfsuite
import json
import sys

def word_features(sentence_list, index):
    """ Extract features from words """

    word = sentence_list[index][0]
    features = [
        'Word = ' + word.lower(),
        'Word last 3 letter     = ' + word[-3:],
        'Word last 2 letter     = ' + word[-2:],
        'Word is in uppercase   = %s' % word.isupper(),
        'Word is a title        = %s' % word.istitle(),
        'Word is a number       = %s' % word.isdigit()
    ]

    if index > 0:
        word_previous = sentence_list[index-1][0]
        postag_previous = sentence_list[index-1][1]
        features.extend([
            'Previous word                  = ' + word_previous.lower(),
            'Previous word is a title       = %s' % word_previous.istitle(),
            'Previous word is in uppercase  = %s' % word_previous.isupper()
        ])
    else:
        features.append('Start')
        
    if index < len(sentence_list)-1:
        word_next = sentence_list[index+1][0]
        postag_next = sentence_list[index+1][1]
        features.extend([
            'Next word              = ' + word_next.lower(),
            'Next word is title     = %s' % word_next.istitle(),
            'Next word is upper     = %s' % word_next.isupper()
        ])
    else:
        features.append('End')

    return features

def sentence_to_words(sentence_list):
    """ Extract words from sentences """
    return [word_features(sentence_list, index) for index in range(len(sentence_list))]

def sentence_labels(sentence_list):
    """ Extract label from sentences """
    return [postag for token, postag in sentence_list]

def train_post_tag(data_dir, model_name):
    """ train data for postagging """
    with open(data_dir) as json_data:
        data = json.load(json_data)
        sentence = [data[i]['pos_tag'] for i in range(len(data))]

    x_train = [sentence_to_words(sentence[i]) for i in range(len(sentence))]
    y_train = [sentence_labels(sentence[i]) for i in range(len(sentence))]

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

    trainer.train(model_name)

if __name__ == '__main__':
    train_post_tag(sys.argv[1], sys.argv[2])
    tagger = pycrfsuite.Tagger()
    tagger.open(sys.argv[2])
    example_sent = [['string.upper','java'],['(','EX'], [")", "EX"]]

    print("Predicted:", ' '.join(tagger.tag(sentence_to_words(example_sent))))
    print("Correct:  ", ' '.join(sentence_labels(example_sent)))
