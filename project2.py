# trains and tests on same csv file

import numpy as np
# import pandas as pd
import csv,re,string
from sklearn import svm
import glob,os

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

    return feature_matrix

def train(features, labels):
    model = svm.LinearSVC(penalty="l2", loss="hinge", dual=True, C=1, random_state=486, max_iter=10000)
    model.fit(features, labels)
    return model

def predict(model, test_data):
    return model.predict(test_data)

def main():
    # X: training data
    # Y: testing data
    X_reviews = []  #2d list
    Y_reviews = []
    X_labels = []     #1d list
    Y_labels = []

    # get list of files to train and test
    training_files = glob.glob(os.path.join("training/", "*.csv"))
    vocab = {}

    reviews = []
    labels = []

    # preprocess training data
    for training_file in training_files:
        with open(training_file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) != 3:
                    continue
                my_string = preprocessing(row[0])
                for s in my_string:
                    if s not in vocab: vocab[s] = len(vocab)
                reviews.append(my_string)
                labels.append(row[2])

    num_reviews = len(reviews)
    X_reviews = reviews[:int(2*num_reviews/3)]
    Y_reviews = reviews[-int(num_reviews/3):]
    X_labels = labels[:int(2*num_reviews/3)]
    Y_labels = labels[-int(num_reviews/3):]

    X_labels = np.array(X_labels[1:])
    Y_labels = np.array(Y_labels[1:])

    X_features = generate_feature_matrix(X_reviews[1:], vocab)
    Y_features = generate_feature_matrix(Y_reviews[1:], vocab)
    # train classifier
    classifier = train(X_features, X_labels)
    # test classifier
    test_predict = predict(classifier, Y_features)
    # evaluate
    # compare Y_labels with test_predict
    # print(len(Y_labels), len(test_predict))
    correct = 0
    for i in range(0, len(Y_labels)):
        print(Y_labels[i] + ", " + test_predict[i])
        if Y_labels[i] == test_predict[i]:
            correct += 1
    print("accuracy: ", correct/len(Y_labels) * 100, "%")

if __name__ == "__main__":
    main()

