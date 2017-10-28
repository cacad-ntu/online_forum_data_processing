from classifier.my_count_vectorizer import MyCountVectorizer
from sklearn.model_selection import train_test_split
from classifier.rnn import RNN
from classifier.bayes_classifier import Classifier
import json
from sklearn.preprocessing import OneHotEncoder

"""
    -   Deducing the negative sentence can be easily done by using Bayes Inference,
    -   Deducing whether the sentence talks about error is quite hard problem, since
        each token is not independent
    -   Deducing the semantic of sentence has the same complexity
        with the above application
        
        "negation_label": 1,
        "error_label": 0,
        "semantic_label": 0
"""

data_vector = MyCountVectorizer()


def init_application():
    data_x = []

    data_y_error = []
    data_y_semantic = []
    data_y_negative = []

    with open("./data/data_class.json") as test:
        datas = json.load(test)

    for data in datas:
        data_x.append(data["pos"])

        # RNN
        data_y_error.append([data["error_label"]])
        data_y_semantic.append([data["semantic_label"]])

        # Naive Bayes
        data_y_negative.append([data["negation_label"]])

    data_x = data_vector.fit_transform(data_x)

    apps_err = prepare_error_application(data_x, data_y_error)
    apps_neg = prepare_negative_application(data_x, data_y_negative)
    apps_sem = prepare_semantic_application(data_x, data_y_semantic)
    return apps_err, apps_neg, apps_sem


def prepare_negative_application(data_x, data_y_negative):

    classifier = Classifier()
    data_x = classifier.tokenize(data_x)

    x_train_negative, x_test_negative, \
        y_train_negative, y_test_negative = train_test_split(data_x, data_y_negative, test_size=0.33, random_state=42)

    print "Start Training Negative Application"
    classifier.start_train(x_train=x_train_negative, y_train=y_train_negative)
    print "Finish Training the Negative Application"
    classifier.start_prediction(x_test=x_test_negative, y_test=y_test_negative)
    return classifier

    # onehot_encoder = OneHotEncoder(sparse=False)
    # data_y_negative = onehot_encoder.fit_transform(data_y_negative)
    #
    # x_train_negative, x_test_negative, \
    #     y_train_negative, y_test_negative = train_test_split(data_x, data_y_negative, test_size=0.33, random_state=42)
    #
    # rnn = RNN(x_train=x_train_negative, y_train=y_train_negative,
    #           x_test=x_test_negative, y_test=y_test_negative,
    #           max_features=len(x_train_negative[0]), num_neurons=200
    #           )
    #
    # rnn.start_train(batch_size=28, epochs=2)
    # return rnn


def prepare_semantic_application(data_x, data_y_semantic):

    onehot_encoder = OneHotEncoder(sparse=False)
    data_y_semantic = onehot_encoder.fit_transform(data_y_semantic)

    x_train_semantic, x_test_semantic, \
        y_train_semantic, y_test_semantic = train_test_split(data_x, data_y_semantic, test_size=0.33, random_state=42)

    rnn = RNN(x_train=x_train_semantic, y_train=y_train_semantic,
              x_test=x_test_semantic, y_test=y_test_semantic,
              max_features=len(x_train_semantic[0])
              )

    rnn.start_train(batch_size=28, epochs=1)
    return rnn


def prepare_error_application(data_x, data_y_error):
    onehot_encoder = OneHotEncoder(sparse=False)
    data_y_error = onehot_encoder.fit_transform(data_y_error)

    x_train_error, x_test_error, \
        y_train_error, y_test_error = train_test_split(data_x, data_y_error, test_size=0.33, random_state=42)

    rnn = RNN(x_train=x_train_error, y_train=y_train_error,
              x_test=x_test_error, y_test=y_test_error,
              max_features=len(x_train_error[0])
              )

    rnn.start_train(batch_size=28, epochs=1)
    return rnn

if __name__ == "__main__":
    apps_err, apps_neg, apps_sem = init_application()

    opt = raw_input("input 1) for error application, 2) negative application, 3) semantic application, 4) exit: \n")

    while opt != 4:

        sentence = str(raw_input("Please input the sentence here: \n"))

        sentence_trans = data_vector.transform_individual_sentence(sentence)

        if opt == '1':

            prediction = apps_err.predict_one_data([sentence_trans])

            if prediction[0]:
                print 'it is an error sentence'
            else:
                print 'it is not an error sentence'
        elif opt == '2':

            prediction = apps_neg.start_predict_one([sentence_trans])
            print "prediction: %s \n" % prediction

            if prediction[0]:
                print 'it is a negative application'
            else:
                print 'it is not a negative application'

        elif opt == '3':

            prediction = apps_sem.predict_one_data([sentence_trans])

            if prediction == 0:
                print 'it is neutral application'
            elif prediction == 1:
                print 'it is a positive application'
            else:
                print 'it is a negative application'
        elif opt == '4':
            break

        opt = raw_input("input 1) for error application, 2) negative application, 3) semantic application, 4) exit \n")