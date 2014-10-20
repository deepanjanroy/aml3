# You can get logs by using --logfile="foo.py" switch in ipython. I'm in love with this feature.
# It's easy then to clean it up a bit and convert it to real code.

import sys
sys.path.insert(0, "..")

import csvio
X = csvio.load_csv_data("../../data/train_inputs.csv")
X_train, X_test = X[:40000], X[40000:]
y = csvio.load_csv_data("../../data/train_outputs.csv", data_type=int)
y_train, y_test = y[:40000], y[40000:]



from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
# 0.24728888

from sklearn import linear_model
clf = linear_model.SGDClassifier()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
# 0.2051

from sklearn import neighbors
clf = neighbors.KNeighborsClassifier(1, weights='distance')
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
# 0.2303

clf = neighbors.KNeighborsClassifier(15, weights='distance')
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
# 0.13869999

from sklearn import svm
clf = svm.SVC()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
# 0.362300000000000001

clf = svm.SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
# 0.3110999


# Kaggle submission
clf = svm.SVC()
clf.fit(X, y)
kaggle_X = dp.load_csv_data("../../data/test_inputs.csv")
preds = clf.predict(kaggle_X)
csvio.write_csv_output(preds, "first_submission.csv")

