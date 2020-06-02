import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

path = "../Data"


def multilabel_confusion_matrix(predict):
    test_label = [i.strip().split() for i in open(path + "/test-label.dat").readlines()]

    for i in range(len(test_label)):
        item = test_label[i]
        for j in range(len(item)):
            item[j] = int(item[j])

        test_label[i] = item

    labels = []

    for item in open(path + "/labels.txt").readlines():
        item = item.replace('\n', '')
        label_name, _ = item.split(', ')
        labels.append(label_name)

    y_true = np.array(test_label)
    y_pred = np.array(predict)

    conf_mat_dict = {}

    for label_col in range(len(labels)):
        y_true_label = y_true[:, label_col]
        y_pred_label = y_pred[:, label_col]
        conf_mat_dict[labels[label_col]] = confusion_matrix(y_pred=y_pred_label, y_true=y_true_label)

    for label, matrix in conf_mat_dict.items():
        print("Confusion matrix for label {}:".format(label))
        print(matrix)
    return


def read_train(path):
    with open(path) as f:
        line = f.readline()
        docs = []
        docs_bin = []
        while line:
            line = line.rstrip('\n').split()
            L = []
            for i in line:
                try:
                    L.append(int(i))
                except:
                    continue
            docs.append(np.array(L))

            words_bin = np.zeros(8520)
            for i in L:
                words_bin[i] += 1

            docs_bin.append(words_bin)
            line = f.readline()
        docs = np.array(docs)
        docs_bin = np.array(docs_bin)
    return docs_bin, docs


def tfIdf(docs_bin):
    tf_transformer = TfidfTransformer(use_idf=True).fit(docs_bin)
    X_train_tf = tf_transformer.transform(docs_bin)
    return X_train_tf


def main():
    docs_bin, docs = read_train(path + "/train-data.dat")
    X_train_tfidf = tfIdf(docs_bin)

    train_label = [i.strip() for i in open(path + "/train-label.dat").readlines()]
    train_label = np.array(train_label)
    clf = LinearSVC().fit(X_train_tfidf, train_label)

    test_docs_bin, test_docs = read_train(path + "/test-data.dat")
    X_test_tfidf = tfIdf(test_docs_bin)

    predicted = clf.predict(X_test_tfidf)

    predict = []
    for item in predicted:
        result = item.split()
        for r in range(len(result)):
            result[r] = int(result[r])
        predict.append(result)

    multilabel_confusion_matrix(predict)
    return


if __name__ == "__main__":
    main()
