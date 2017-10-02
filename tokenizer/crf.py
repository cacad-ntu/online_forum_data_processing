from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite

def word_features(sentence_list, word):
    word = sentence_list[word][0]
    postag = sentence_list[word][1]
    features = [
        'Word =' + word.lower(),
        'Word last 3 letter     = ' + word[-3:],
        'Word last 2 letter     = ' + word[-2:],
        'Word is in uppercase   = %s' % word.isupper(),
        'Word is a title        = %s' % word.istitle(),
        'Word is a number       = %s' % word.isdigit(),
        'POS tag                = ' + postag
    ]

    if word > 0:
        word_previous = sentence_list[word-1][0]
        postag_previous = sentence_list[word-1][1]
        features.extend([
            'Previous word                  = ' + word_previous.lower(),
            'Previous word is a title       = %s' % word_previous.istitle(),
            'Previous word is in uppercase  = %s' % word_previous.isupper(),
            'Previous word\'s POS tag       = ' + postag_previous 
        ])
    else:
        features.append('Start')
        
    if word < len(sentence_list)-1:
        word_next = sentence_list[word+1][0]
        postag_next = sentence_list[word+1][1]
        features.extend([
            'Next word              = ' + word_next.lower(),
            'Next word is title     = %s' % word_next.istitle(),
            'Next word is upper     = %s' % word_next.isupper(),
            'Next word\'s POS tag   = ' + postag_next,
        ])
    else:
        features.append('End')
                
    return features

def sentence_to_words(sentence_list):
    return [word_features(sentence_list, word) for word in range(len(sentence_list))]

def sentence_labels(sentence_list):
    return [label for token, postag, label in sentence_list]

def sent2tokens(sentence_list):
    return [token for token, postag, label in sentence_list]

train_data = []
test_data = []

X_train = [sentence_to_words(data) for data in train_data]
y_train = [sentence_labels(data) for data in train_data]

X_test = [sentence_to_words(data) for data in test_data]
y_test = [sentence_labels(data) for data in test_data]

trainer = pycrfsuite.Trainer(verbose=False)

for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

# trainer.set_params({
#     'c1': 1.0,   # coefficient for L1 penalty
#     'c2': 1e-3,  # coefficient for L2 penalty
#     'max_iterations': 50,  # stop earlier

#     # include transitions that are possible, but not observed
#     'feature.possible_transitions': True
# })

# trainer.train('conll2002-esp.crfsuite')