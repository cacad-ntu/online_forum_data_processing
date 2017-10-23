from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from MyCountVectorizer import MyCountVectorizer


class Classifier:

    def __init__(self, data):
        self.data = data
        return

    def tokenize(self, data):
        print ("start the downscaling for optimization \n")
        tfidf_transformer = TfidfTransformer()
        my_count_vect = MyCountVectorizer()
        X_train_counts = my_count_vect.fit_transform(data)
        self.X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    def start_train(self, x_train, y_train):

        self.tokenize(x_train)
        print ("start the downscaling for optimization \n")
        self.clf = MultinomialNB().fit(self.X_train_tfidf, y_train)

    def start_prediction(self, x_test, y_test):

        print accuracy_score(self.clf.predict(x_test), y_test)
