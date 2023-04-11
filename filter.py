# removes reviews from csv file until num helpful == num unhelpful

import csv,re,string
def main():
    training_file = "testing/amazon_dataset.csv"
    output_file = "testing/amazon_dataset_balanced.csv"

    num_helpful = 0
    num_unhelpful = 0
    reviews = []
    with open(training_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            reviews.append(row)
            if row[2] == '1':
                num_helpful += 1
            elif row[2] == '-1':
                num_unhelpful += 1
    
    # count the unhelpful reviews until counter > num_helpful
    counter = 0
    with open(output_file, "w") as output:
        for review in reviews:
            if review[2] == "1" or counter < num_helpful:
                output.write(review[0] + "," + review[1] + "," + review[2] + "\n")
            if review[2] == "-1":
                counter += 1


if __name__ == "__main__":
    main()
