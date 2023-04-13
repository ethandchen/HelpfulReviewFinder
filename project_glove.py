import numpy as np
import csv,re,string
from sklearn import svm
from sklearn.decomposition import PCA
import glob,os
from matplotlib import pyplot as plt

def preprocess_glove_embeddings(embedding_dim):
    word_to_index = {}    #0: padding, 1: UNKA
    glove_embeddings = []
    #add first two entries of random numbers for PAD and UNKA
    zero_embedding = []
    for a in range(embedding_dim):
        zero_embedding.append(0)
    glove_embeddings.append(zero_embedding)
    rand_embedding = []
    for a in range(embedding_dim):
        rand_embedding.append(np.random.normal())
    glove_embeddings.append(rand_embedding)

    glove_file = "glove.6B.50d.txt"
    glove_path = os.path.join(os.getcwd(), "glove.6B", glove_file)
    with open(glove_path, "r", encoding="utf-8") as reader:
        index = 2
        for line in reader:
            glove_line = line
            glove_list = glove_line.split()
            word = glove_list[0]
            word_to_index[word] = index
            embedding = [float(e) for e in glove_list[1:]]
            glove_embeddings.append(embedding)
            index += 1
    return word_to_index, glove_embeddings

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


def generate_feature_matrix(list_of_tokens, word_to_index, glove_embeddings):
    number_of_reviews = len(list_of_tokens)
    max_dim = 10000
    feature_matrix = np.zeros((number_of_reviews, max_dim))
    for i, review in enumerate(list_of_tokens):
        counter = 0
        for token in review:
            if token in word_to_index:
                ind = word_to_index[token]
                embedding = glove_embeddings[ind]
                for k, num in enumerate(embedding):
                    feature_matrix[i][counter] = embedding[k]
                    counter += 1
            if counter > max_dim-51:
                break
    pca_model = PCA(n_components=300)
    pca_model.fit(feature_matrix)
    pca_model.fit(feature_matrix)
    pca_mat = pca_model.transform(feature_matrix)

    return pca_mat

def train(features, labels):
    model = svm.LinearSVC(penalty="l2", loss="hinge", dual=True, C=0.01, random_state=486, max_iter=5000)
    model.fit(features, labels)
    return model

def predict(model, test_data):
    return model.predict(test_data)

def run(file):
    # X: training data
    # Y: testing data
    X_reviews = []  #2d list
    Y_reviews = []
    X_labels = []     #1d list
    Y_labels = []

    reviews = []
    labels = []

    # preprocess glove embeddings
    word_to_index, glove_embeddings = preprocess_glove_embeddings(50)

    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) != 3:
                continue
            my_string = preprocessing(row[0])
            reviews.append(my_string)
            labels.append(row[2])
    
    num_reviews = len(reviews)
    X_reviews = reviews[:int(2*num_reviews/3)]
    Y_reviews = reviews[-int(num_reviews/3):]
    X_labels = labels[:int(2*num_reviews/3)]
    Y_labels = labels[-int(num_reviews/3):]

    X_labels = np.array(X_labels[1:])
    Y_labels = np.array(Y_labels[1:])

    X_features = generate_feature_matrix(X_reviews[1:], word_to_index, glove_embeddings)
    Y_features = generate_feature_matrix(Y_reviews[1:], word_to_index, glove_embeddings)
    # train classifier
    classifier = train(X_features, X_labels)
    # test classifier
    test_predict = predict(classifier, Y_features)
    # evaluate
    correct = 0
    for i in range(0, len(Y_labels)):
        if Y_labels[i] == test_predict[i]:
            correct += 1
    accuracy = correct/len(Y_labels) * 100
    print("accuracy: ", accuracy, "%")
    return accuracy
    

def main():
    # get list of training files
    training_files = glob.glob(os.path.join("balanced/", "*.csv"))

    # run on each file
    accuracies = []
    for training_file in training_files:
        accuracy = run(training_file)
        accuracies.append(accuracy)
    file_names = ["Amazon", "Walmart", "Yelp"]
    plt.bar(file_names, accuracies)
    plt.xlabel("Training File")
    plt.ylabel("Accuracy %")
    plt.title("Performance of Linear SVM Model using Glove Embeddings")
    plt.savefig("glove_results.jpg")
    plt.close()

if __name__ == "__main__":
    main()

