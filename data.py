# imports and declarations
import csv
results = []

# function designed to prepare features
def features():
    with open("/home/ahsan/PycharmProjects/lrrk2_vs_estimator/a/main2(features).csv") as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        for row in reader:  # each row is a list
            results.append(row)
    return results

# function designed to prepare labels
def labels():
    # convert csv target column to coma sepreated form online and save
    # it in .txt than importing that .txt here to save as a simple list.
    f = open("/home/ahsan/PycharmProjects/lrrk2_vs_estimator/a/labels.txt", "r")
    x = f.readline()
    x = x.split(',')
    return x


if __name__ == '__main__':
    print("data")
