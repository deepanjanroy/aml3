# File meant to be used with IPython.parallel

import sys
sys.path.append("..")

import csvio
import logistic_reg as lg
import random
import numpy as np
from sklearn.decomposition import PCA

X_all = np.array(csvio.load_csv_data("../../data/train_inputs.csv", max_rows=20000))  # Size 50000
y_all = np.array(csvio.load_csv_data("../../data/train_outputs.csv", max_rows=20000))

X_train = X_all[:40000]
y_train = y_all[:40000]

X_test = X_all[40000:]
y_test = y_all[40000:]

cv_data = []
def process_data(fold):
    train_indices = range(0, fold*10000) + range((fold + 1) * 10000, 40000)
    val_indices = range(fold * 10000, (fold + 1) * 10000)

    pca = PCA(n_components=100)

    cv_X_train = pca.fit_transform(X_train[train_indices])
    cv_X_val = pca.transform(X_train[val_indices])

    cv_y_train = y_train[train_indices]
    cv_y_val = y_train[val_indices]

    cv_data.append((cv_X_train, cv_y_train, cv_X_val, cv_y_val))


def cross_validate(fold_no, fit_kwargs={}):
    cv_X_train, cv_y_train, cv_X_val, cv_y_val = cv_data[0]
    train_indices = range(0, fold*10000) + range((fold + 1) * 10000, 40000)
    val_indices = range(fold * 10000, (fold + 1) * 10000)

    cv_X_train = X_train[train_indices]
    cv_X_val = X_train[val_indices]

    cv_y_train = y_train[train_indices]
    cv_y_val = y_train[val_indices]

    clf = lg.MulticlassLogisticRegressor()
    clf.fit(cv_X_train, cv_y_train, **fit_kwargs)
    score = clf.score(cv_X_val, cv_y_val)

    randint= random.randint(0, 100000000000000000)
    fname = "cvlogs_{0}".format(randint)
    with open(fname, "w") as f:
        f.write(str(fit_kwargs))
        f.write(score)

    return score


