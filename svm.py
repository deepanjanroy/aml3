import csvio
import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm

X = csvio.load_csv_data("../data/train_inputs.csv")
X = np.array(X)
y = csvio.load_csv_data("train_outputs.csv")

pca = PCA(n_components=500)
pcaX = pca.fit_transform(X)
X_kaggle = csvio.load_csv_data("test_inputs.csv")
X_kaggle = np.array(X_kaggle)
pcaKaggle = pca.transform(X_kaggle)

lin_clf = svm.LinearSVC()
pred = lin_clf.predict(pcaKaggle)
csvio.write_csv_output([int(x) for x in pred], "svm_test.csv")
