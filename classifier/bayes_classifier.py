from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from tokenizer.regex import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


class Classifier:

    def __init__(self):
        self.clf = None
        self.clf_svm = None

    def start_train_pipeline(self, data_x_train, data_y_train, data_x_test, data_y_test):

        myTokenizer = Tokenizer()
        vectorizer = CountVectorizer(tokenizer=lambda text: myTokenizer.start_tokenize(text, True))

        text_clf = Pipeline([('vect', vectorizer),
                             ('tfidf', TfidfTransformer()),
                             ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)),
                             ])
        self.clf = text_clf.fit(data_x_train, data_y_train)

        predicted = self.clf.predict(data_x_test)
        data_y_test = np.array(data_y_test).reshape(len(data_y_test))

        print "accuracy: %s \n" % accuracy_score(predicted, data_y_test)
        print "f1: %s \n" % f1_score(predicted, data_y_test, average="macro")
        print "precision: %s \n" % precision_score(predicted, data_y_test, average="macro")
        print "recall: %s \n" % recall_score(predicted, data_y_test, average="macro")

    def start_predict_one(self, x_test):

        predict = self.clf.predict(np.array(x_test))
        return predict
