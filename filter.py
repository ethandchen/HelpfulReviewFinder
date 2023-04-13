# removes reviews from csv file until num helpful == num unhelpful

import csv,re,string,random
def main():
    training_file = "training/yelp-reviews.csv"
    output_file = "balanced/yelp-reviews_balanced.csv"

    helpful_reviews = []
    unhelpful_reviews = []
    with open(training_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[2] == '1':
                helpful_reviews.append(row)
            elif row[2] == '-1':
                unhelpful_reviews.append(row)
    if len(helpful_reviews) > len(unhelpful_reviews):
        helpful_reviews = helpful_reviews[:len(unhelpful_reviews)]
    else:
        unhelpful_reviews = unhelpful_reviews[:len(helpful_reviews)]
    
    reviews = helpful_reviews + unhelpful_reviews
    random.shuffle(reviews)
    with open(output_file, "w") as output:
        for review in reviews:
            output.write(review[0] + "," + review[1] + "," + review[2] + "\n")


if __name__ == "__main__":
    main()
