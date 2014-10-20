import csv

def load_csv_data(file_path, data_type=float, max_rows=None):
    """
    Example:
    >>> csvio.load_csv_data("train_input.csv")
    # Loads all train_input and returns a list of lists

    >>> csvio.load_csv_data("train_output.csv" data_type=int)
    # Reads csv numbers as integers instead of floats
    # Also, there is only one column (excluding id column), it flattens
    [[2], [3], [4]] into [2, 3, 4]

    >>> csvio.load_csv_data("train_input.csv", max_rows=1000)
    # Only loads 1000 rows. Great for quick testing
    """

    data = []

    with open(file_path) as f:

        reader = csv.reader(f)
        reader.next()  # Getting rid of the headers
        data = []
        for i, row in enumerate(reader, 1):
            if max_rows is not None and i > max_rows:
                break
            row_sans_id = row[1:]
            converted_row = [data_type(x) for x in row_sans_id]
            if len(converted_row) == 1:
                # We want [4, 5, 6] instead of [[4], [5], [6]]
                data.append(converted_row[0])
            else:
                data.append(converted_row)

    return data


def write_csv_output(predictions, file_path):
    """
    Example:
    >>> csvio.write_csv_output(clf.predict(X_test), "deep_learning_submission.csv")
    """

    out_file = open(file_path, "wb")
    writer = csv.writer(out_file, delimiter=',')
    writer.writerow(['Id', 'Prediction'])  # header
    for row in enumerate(predictions, 1):
        writer.writerow(row)
    out_file.close()
