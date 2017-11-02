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
        features.extend([
            'Previous word                  = ' + word_previous.lower(),
            'Previous word is a title       = %s' % word_previous.istitle(),
            'Previous word is in uppercase  = %s' % word_previous.isupper()
        ])
    else:
        features.append('Start')
        
    if index < len(sentence_list)-1:
        word_next = sentence_list[index+1][0]
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
    """ Extract labels from sentences """
    return [postag for token, postag in sentence_list]

def train_post_tag(data_dir, k_fold, model_name):
    """ train data for postagging """
    with open(data_dir) as json_data:
        data = json.load(json_data)
        sentence = [data[i]['pos_tag'] for i in range(len(data))]
     
    if k_fold == 1:
        x_train = [sentence_to_words(sentence[i]) for i in range(len(sentence))]
        y_train = [sentence_labels(sentence[i]) for i in range(len(sentence))]

        trainer = pycrfsuite.Trainer()

        for xdata, ydata in zip(x_train, y_train):
            trainer.append(xdata, ydata)

        trainer.set_params({
            'c1': 1.0,   # coefficient for L1 penalty
            'c2': 1e-3,  # coefficient for L2 penalty
            'max_iterations': 100,  # stop earlier

            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })

        trainer.train(model_name + '.crfsuite')

        return (model_name + '.crfsuite')

    accuracy = 0
    model_num = None
    accuracy_list = []  

    for i in range (k_fold):
        temp_accuracy = 0
        word_count = 0

        if (i == 0):
            x_train = [sentence_to_words(sentence[j]) for j in range((i+1)*len(sentence)//k_fold, len(sentence))]
            y_train = [sentence_labels(sentence[j]) for j in range((i+1)*len(sentence)//k_fold, len(sentence))]

        elif (i == (k_fold-1)):
            x_train = [sentence_to_words(sentence[j]) for j in range(i*len(sentence)//k_fold)]
            y_train = [sentence_labels(sentence[j]) for j in range(i*len(sentence)//k_fold)]
        else:
            x_train = [sentence_to_words(sentence[j]) for j in range(i*len(sentence)//k_fold)]
            y_train = [sentence_labels(sentence[j]) for j in range(i*len(sentence)//k_fold)]

            x_train.extend(sentence_to_words(sentence[j]) for j in range((i+1)*len(sentence)//k_fold, len(sentence)))
            y_train.extend(sentence_labels(sentence[j]) for j in range((i+1)*len(sentence)//k_fold, len(sentence)))

        x_test = [sentence_to_words(sentence[j]) for j in range(i*len(sentence)//k_fold, (i+1)*len(sentence)//k_fold)]
        y_test = [sentence_labels(sentence[j]) for j in range(i*len(sentence)//k_fold, (i+1)*len(sentence)//k_fold)]
        
        trainer = pycrfsuite.Trainer()

        for xdata, ydata in zip(x_train, y_train):
            trainer.append(xdata, ydata)

        trainer.set_params({
            'c1': 1.0,   # coefficient for L1 penalty
            'c2': 1e-3,  # coefficient for L2 penalty
            'max_iterations': 50,  # stop earlier

            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })

        trainer.train(model_name + str(i) + '.crfsuite')

        tagger = pycrfsuite.Tagger()
        tagger.open(model_name + str(i) + '.crfsuite')

        for token, label in zip(x_test, y_test):
            tagged_data = tagger.tag(token)
            for j in range (len(label)):
                if  tagged_data[j] == label[j]:
                    temp_accuracy += 1
                word_count += 1

        temp_accuracy = temp_accuracy / word_count
        accuracy_list.append(temp_accuracy)
        

        if (temp_accuracy > accuracy):
            accuracy = temp_accuracy
            model_num = i

    print(accuracy_list)
    print(model_num)
    return (model_name + str(model_num) + '.crfsuite')

if __name__ == '__main__':
    tagger = pycrfsuite.Tagger()
    tagger.open(train_post_tag(sys.argv[1], int(sys.argv[2]), sys.argv[3]))
    #tagger.open('test1.crfsuite')

    data = ['you', 'are', 'my', 'friend', '.']

    print("Word     : " +  str(data))
    print("POS tag  : " +  str(tagger.tag(sentence_to_words(data))))
    print("POS tag  : " +  str(tagger.tag(data)))
