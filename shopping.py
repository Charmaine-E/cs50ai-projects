import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from CSV file 
    """
    months = ["jan", "feb", "mar", "apr", "may", "jun",
              "jul", "aug", "sep", "oct", "nov", "dec"]
    int_cols = {0, 2, 4, 11, 12, 13, 14}
    float_cols = {1, 3, 5, 6, 7, 8, 9}

    evidence = []
    labels = []
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if not row:
                continue
            values = []
            for i in range(17):
                cell = row[i].strip()
                if i in int_cols:
                    values.append(int(cell))
                elif i in float_cols:
                    values.append(float(cell))
                elif i == 10:
                    values.append(months.index(cell[:3].lower()))
                elif i == 15:
                    values.append(1 if cell == "Returning_Visitor" else 0)
                else:  # Weekend
                    values.append(1 if cell.upper() == "TRUE" else 0)
            evidence.append(values)
            labels.append(1 if row[17].strip().upper() == "TRUE" else 0)
    return evidence, labels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    
    true_pos = sum(1 for l, p in zip(labels, predictions) if l == 1 and p == 1)
    true_neg = sum(1 for l, p in zip(labels, predictions) if l == 0 and p == 0)
    positives = sum(1 for l in labels if l == 1)
    negatives = sum(1 for l in labels if l == 0)
    return true_pos / positives, true_neg / negatives


if __name__ == "__main__":
    main()