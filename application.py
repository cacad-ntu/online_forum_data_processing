from classifier.Classifier import Classifier
from sklearn.model_selection import train_test_split
import json


if __name__ == "__main__":

    data_x = []
    data_y = []

    classifier = Classifier()

    with open("./data/pos_tag_all_class.json") as test:
        datas = json.load(test)

    for data in datas:
        data_x.append(data["origin"])
        data_y.append(data["label"])

    data_x = classifier.tokenize(data_x)

    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.33, random_state=42)

    classifier.start_train(x_train=X_train, y_train=y_train)
    classifier.start_prediction(X_test, y_test)