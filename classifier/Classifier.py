from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


class Classifier:

    def __init__(self, data):
        self.data = data
        return

    def do_scikit_tokenization(self, data):
        count_vect = CountVectorizer()
        self.data = count_vect.fit_transform(data)

    def start_train(self, target):

        print ("start the downscaling for optimization \n")
        tf_transformer = TfidfTransformer(use_idf=False).fit(self.data)
        X_train = tf_transformer.transform(self.data)

        print ("start the learning")
        clf = MultinomialNB().fit(X_train, target)