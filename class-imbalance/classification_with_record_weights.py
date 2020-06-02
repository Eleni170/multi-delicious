from costcla.metrics import cost_loss
from costcla.models import CostSensitiveLogisticRegression

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer

import numpy as np


def read_train(path):
    with open(path) as f:
        line = f.readline()
        docs_bin = []
        while line:
            line = line.rstrip('\n').split()
            L = []
            for i in line:
                try:
                    L.append(int(i))
                except:
                    continue

            words_bin = np.zeros(8520)
            for i in L:
                words_bin[i] += 1

            docs_bin.append(words_bin)
            line = f.readline()
        docs_bin = np.array(docs_bin)
    return docs_bin


def tfIdf(docs_bin):
    tf_transformer = TfidfTransformer(use_idf=True).fit(docs_bin)
    X_tf = tf_transformer.transform(docs_bin)
    return X_tf.toarray()


def load_labels(path, class_index):
    y_list = [i.strip().split() for i in open(path).readlines()]

    for i in range(len(y_list)):
        item = y_list[i]
        for j in range(len(item)):
            item[j] = int(item[j])

        y_list[i] = item[class_index]

    return np.array(y_list)


def calculate_cost_matrix(y_list):
    y_list_len = len(y_list)
    percentage_0 = np.count_nonzero(y_list == 0) / y_list_len
    percentage_1 = np.count_nonzero(y_list == 1) / y_list_len

    cost_matrix = []
    for i in range(len(y_list)):
        cost_matrix.append([1 / (percentage_0 * y_list_len), 1 / (percentage_1 * y_list_len), 0, 0])

    return np.array(cost_matrix)


def train(class_index):
    docs_bin = read_train("../Data/train-data.dat")
    X_train = tfIdf(docs_bin)

    y_train = load_labels("../Data/train-label.dat", class_index)

    cost_mat_train = calculate_cost_matrix(y_train)

    f = CostSensitiveLogisticRegression()
    f.fit(X_train, y_train, cost_mat_train)
    return f


def predict(f, class_index):
    test_docs_bin = read_train("../Data/test-data.dat")
    X_test = tfIdf(test_docs_bin)

    y_test = load_labels("../Data/test-label.dat", class_index)

    cost_mat_test = calculate_cost_matrix(y_test)

    y_pred_test_cslr = f.predict(X_test)

    return cost_loss(y_test, y_pred_test_cslr, cost_mat_test)


def main():
    classes = []

    for item in open("../Data/labels.txt").readlines():
        item = item.replace('\n', '')
        label_name, _ = item.split(', ')
        classes.append(label_name)

    for class_index in range(len(classes)):
        f = train(class_index)
        cl = predict(f, class_index)
        print("-----")
        print("Class: " + classes[class_index])
        print("Cost Loss: " + str(cl))
        print("-----")


if __name__ == "__main__":
    main()
