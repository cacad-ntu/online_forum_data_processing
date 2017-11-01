from classifier.my_count_vectorizer import MyCountVectorizer
from sklearn.model_selection import train_test_split
from classifier.rnn import RNN
from classifier.bayes_classifier import Classifier
import json
from sklearn.preprocessing import OneHotEncoder
from classifier.regex_checker import regex_checking

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

    buffer_data_x = data_x
    data_x = data_vector.fit_transform(data_x)
    use_rnn = False

    # apps_err = prepare_error_application(data_x if use_rnn else buffer_data_x, data_y_error, use_rnn=use_rnn)
    print '----------------FINISH ERROR TRAINING ----------------- \n'
    apps_err = None
    # apps_neg = prepare_negative_application(buffer_data_x, data_y_negative)
    apps_neg = None
    # apps_sem = None
    print '----------------FINISH NEGATIVE TRAINING ----------------- \n'
    apps_sem = prepare_semantic_application(data_x if use_rnn else buffer_data_x, data_y_semantic, use_rnn=use_rnn)
    print '----------------FINISH SEMANTIC TRAINING ----------------- \n'
    return apps_err, apps_neg, apps_sem


def prepare_negative_application(data_x, data_y_negative):

    classifier = Classifier()

    x_train_negative, x_test_negative, \
        y_train_negative, y_test_negative = train_test_split(data_x, data_y_negative, test_size=0.33, random_state=42)

    classifier.start_train_pipeline(x_train_negative, y_train_negative,
                                    x_test_negative, y_test_negative, use_tune=True, use_naive_bayes=True)
    return classifier


def prepare_semantic_application(data_x, data_y_semantic, use_rnn=True):

    onehot_encoder = OneHotEncoder(sparse=False)

    if use_rnn:
        data_y_semantic = onehot_encoder.fit_transform(data_y_semantic)

    x_train_semantic, x_test_semantic, \
        y_train_semantic, y_test_semantic = train_test_split(data_x, data_y_semantic, test_size=0.33, random_state=42)

    if use_rnn:
        classifier = RNN(x_train=x_train_semantic, y_train=y_train_semantic,
                  x_test=x_test_semantic, y_test=y_test_semantic,
                  max_features=len(x_train_semantic[0])
                  )

        classifier.start_train(batch_size=28, epochs=5)
    else:
        classifier = Classifier()
        classifier.start_train_pipeline(x_train_semantic, y_train_semantic,
                                        x_test_semantic, y_test_semantic,
                                        use_tune=True, use_naive_bayes=False)

    return classifier


def prepare_error_application(data_x, data_y_error, use_rnn=True):
    onehot_encoder = OneHotEncoder(sparse=False)

    if use_rnn:
        data_y_error = onehot_encoder.fit_transform(data_y_error)

    x_train_error, x_test_error, \
        y_train_error, y_test_error = train_test_split(data_x, data_y_error, test_size=0.33, random_state=42)

    if use_rnn:
        classifier = RNN(x_train=x_train_error, y_train=y_train_error,
                  x_test=x_test_error, y_test=y_test_error,
                  max_features=len(x_train_error[0])
                  )

        classifier.start_train(batch_size=28, epochs=5)
    else:
        classifier = Classifier()
        classifier.start_train_pipeline(x_train_error, y_train_error,
                                        x_test_error, y_test_error,
                                        use_tune=True, use_naive_bayes=False)

    return classifier

if __name__ == "__main__":
    apps_err, apps_neg, apps_sem = init_application()

    opt = raw_input("input\n 1) for error application\n 2) negative application\n "
                    "3) semantic application\n 4) Negative application using Regex \n 5) Exit. \n")

    while opt != 4:

        sentence = str(raw_input("Please input the sentence here: \n"))

        sentence_trans = data_vector.transform_individual_sentence(sentence)

        if opt == '1':

            if hasattr(apps_err, 'predict_one_data'):
                prediction = apps_err.predict_one_data([sentence_trans])
            else:
                prediction = apps_err.start_predict_one([sentence])

            if prediction[0]:
                print 'it is an error sentence\n'
            else:
                print 'it is not an error sentence\n'
        elif opt == '2':

            prediction = apps_neg.start_predict_one([sentence])
            print "prediction: %s \n" % prediction

            if prediction[0]:
                print 'it is a negative application\n'
            else:
                print 'it is not a negative application\n'

        elif opt == '3':

            if hasattr(apps_sem, 'predict_one_data'):
                prediction = apps_sem.predict_one_data([sentence_trans])
            else:
                prediction = apps_sem.start_predict_one([sentence])

            if prediction == 0:
                print 'it is neutral application\n'
            elif prediction == 1:
                print 'it is a positive application\n'
            else:
                print 'it is a negative application\n'
        elif opt == '4':

            prediction = regex_checking(sentence)
            if prediction:
                print 'it is negative expression\n'
            else:
                print 'it is not a negative application\n'

        elif opt == '5':
            break

        opt = raw_input("input\n 1) for error application\n 2) negative application\n "
                        "3) semantic application\n 4) Negative application using Regex \n 5) Exit. \n")

        if opt == '5':
            break
