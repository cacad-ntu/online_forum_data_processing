from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from MyCountVectorizer import MyCountVectorizer


class Classifier:

    def __init__(self):
        return

    def tokenize(self, data):
        tfidf_transformer = TfidfTransformer()
        my_count_vect = MyCountVectorizer()
        X_train_counts = my_count_vect.fit_transform(data)
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        return X_train_tfidf

    def start_train(self, x_train, y_train):

        self.clf = MultinomialNB().fit(x_train, y_train)

    def start_prediction(self, x_test, y_test):

        predict = self.clf.predict(x_test)

        print "accuracy: %s, recall: %s, precision: %s \n" % (
            accuracy_score(predict, y_test),
            recall_score(predict, y_test, average='weighted'),
            precision_score(predict, y_test, average='weighted')
        )
