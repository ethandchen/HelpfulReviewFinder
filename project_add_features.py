import numpy as np
# import pandas as pd
import csv,re,string
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import glob,os
from matplotlib import pyplot as plt

# preprocess each review
def preprocessing(text):
    text = re.sub('<.*?>', '' , text).lower()
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text).replace('\n', ' ')
    result = []
    with open("stopwords", "r") as f:
        data = f.read().replace('\n', ' ')
        words = data.split(' ')

    for token in text.split(' '):
        if token not in words:
            result.append(token)
    return result


def generate_feature_matrix(list_of_tokens, vocab):
    number_of_reviews = len(list_of_tokens)
    number_of_words = len(vocab)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    for i, reviews in enumerate(list_of_tokens):
        for token in reviews:
            try:
                col = vocab[token]
                feature_matrix[i][col] += 1
            except: print("somethings wrong")
        # feature_matrix[i][-1] = len(reviews)

    return feature_matrix

def train(features, labels, c=1):
    model = svm.LinearSVC(penalty="l2", loss="hinge", dual=True, C=c, random_state=486, max_iter=10000)
    model.fit(features, labels)
    return model

def predict(model, test_data):
    return model.predict(test_data)

def performance(a, b):
    return metrics.accuracy_score(a, b)

def cv_performance(clf, X, y, k=5):
    scores = []
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    for train_index, test_index in skf.split(X, y):
        clf.fit(X[train_index], y[train_index])
        predict = clf.predict(X[test_index])
        scores.append(performance(y[test_index], predict))
    return np.array(scores).mean()

def main():
    # X: training data
    # Y: testing data
    X_reviews = []  #2d list
    Y_reviews = []
    X_labels = []     #1d list
    Y_labels = []

    # get list of training and testing files
    training_files = glob.glob(os.path.join("training/", "*.csv"))
    testing_files = glob.glob(os.path.join("testing/", "*.csv"))
    vocab = {}

    # preprocess training data
    for training_file in training_files:
        with open(training_file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                my_string = preprocessing(row[0])
                for s in my_string:
                    if s not in vocab: vocab[s] = len(vocab)
                X_reviews.append(my_string)
                X_labels.append(row[2])

    # preprocess testing data
    for testing_file in testing_files:
        with open(testing_file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                my_string = preprocessing(row[0])
                for s in my_string:
                    if s not in vocab: vocab[s] = len(vocab)
                Y_reviews.append(my_string)
                Y_labels.append(row[2])

    X_labels = np.array(X_labels[1:])
    Y_labels = np.array(Y_labels[1:])

    X_features = generate_feature_matrix(X_reviews[1:], vocab)
    Y_features = generate_feature_matrix(Y_reviews[1:], vocab)
    # train classifier
    classifier = train(X_features, X_labels, c=0.01)
    # test classifier
    test_predict = predict(classifier, Y_features)
    # evaluate
    # compare Y_labels with test_predict
    # print(len(Y_labels), len(test_predict))
    correct = 0
    for i in range(0, len(Y_labels)):
        # print(test_predict[i])
        if Y_labels[i] == test_predict[i]:
            correct += 1
    print("accuracy: ", correct/len(Y_labels) * 100, "%")
    print('1')
    clf = svm.SVC(kernel="poly", degree=2, C=0.01, gamma="auto")
    clf.fit(X_features, X_labels)
    print(performance(Y_labels, clf.decision_function(Y_features)))

    # c_range = [10,1,0.1, 0.01, 0.001, 0.0001]
    # accuracy = []
    # for c in c_range:
    #     correct = 0
    #     classifier = train(X_features, X_labels, c)
    #     test_predict = predict(classifier, Y_features)
    #     for i in range(0, len(Y_labels)):
    #         if Y_labels[i] == test_predict[i]:
    #             correct += 1
    #     accuracy.append(correct/len(Y_labels) * 100)
    # print(accuracy)
    # plt.plot(c_range, accuracy)
    # plt.xscale("log")
    # plt.xlabel("Value of C hyperparameter")
    # plt.ylabel("Accuracy %")
    # plt.title("Tuning hyperparameter of Linear SVM normal embeddings")
    # plt.savefig("NORMAL.jpg")
    # plt.close()

if __name__ == "__main__":
    main()

