import sys
sys.path.append("..")
import csvio
import logistic_rega as lg
from sklearn.decomposition import PCA

from sklearn.decomposition import PCA
pca = PCA(n_components=100)
new_xs_train = pca.fit_transform(xs_train)
new_xs_test = pca.transform(xs_test)

X_all = csvio.load_csv_data("../../data/train_inputs.csv")
y_all = csvio.load_csv_data("../../data/train_outputs.csv")

