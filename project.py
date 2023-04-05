import numpy as np
import pandas as pd
import csv,re,string
from sklearn import svm
import glob,os


# def load_data(fname):
#     #return 2d matrix of data from csv
#     # in the form of [text, ratings, helpful(1,-1)]
#     text = []
#     ratings = []
#     labels = []
#     with open(fname) as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         for i, row in enumerate(csv_reader):
#             if i == 0 or i == 1: continue
#             for j, data in enumerate(row):
#                 if len(data) != 3: continue
#                 if j == 0:
#                     text.append(data)
#                 elif j == 1:
#                     labels.append(data)
#                 else:
#                     ratings.append(data)
                
#     return text, labels

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

def word_dict(vocab):
    dic = {}
    for i, word in enumerate(vocab):
        dic[word] = i
    return dic

def generate_feature_matrix(list_of_tokens, vocab):
    number_of_reviews = len(list_of_tokens)
    number_of_words = len(vocab)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    for review in list_of_tokens:


def train(features, labels):
    model = svm.LinearSVC(penalty="l2", loss="hinge", dual=True, C=1, random_state=486)
    model.fit(features, labels)
    return model
    
def predict(model, test_data):
    return model.predict(test_data)

def main():
    
    #change the file path to the all data cvs file location
    csv_files = glob.glob(os.path.join("/Users/briansun/Desktop/eecs486/final_project", "*.csv"))
    csv_file_names = [os.path.basename(csv_file) for csv_file in csv_files]
    vocab = set()
    reviews = []  #2d list
    X_labels = []     #1d list
    for file_name in csv_file_names:
        with open(file_name, newline='') as csvfile:
            reader = csv.reader(csvfile)
            #rows = []
            
            for row in reader:
                my_string = preprocessing(row[0])
                for s in my_string:
                    vocab.add(s)
                reviews.append(my_string)
                X_labels.append(row[1])
                # row[0] = my_string
                # rows.append(row)

        # with open(file_name, 'w', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerows(rows)


    # test = load_data('temp1.csv')
    # print(test)
    # # X: training data
    # # Y: testing data
    # X = load_data("data/train.csv")
    # Y = load_data("data/test.csv")



    X_features, X_labels = get_feature_matrix(training_data)
    Y_features, Y_labels = preprocess(testing_data)

    # train classifier
    classifier = train(X_features, X_labels)

    # test classifier
    test_predict = predict(classifier, Y_features)

    # evaluate
    # compare Y_labels with test_predict

if __name__ == "__main__":
    main()

