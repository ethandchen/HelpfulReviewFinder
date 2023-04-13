# HelpfulReviewFinder

This project trains a Linear SVM to determine whether certain reviews are helpful or not helpful.

The file project_tf_features.py trains a linear SVM using each of the three files from the balanced/ folder. The file project.py trains and tests using different csv files, while project_tf_features.py trains and tests on the same file.

The file project_glove.py uses glove embeddings to train the SVM, rather than the term frequency. The glove embeddings can be downloaded from here: https://nlp.stanford.edu/projects/glove/

The file filter.py takes a csv file as input and outputs a new csv file with some of the reviews removed so that the number of helpful reviews and unhelpful reviews are around equal. This helps us train and test using balanced data.

