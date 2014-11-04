from sklearn.decomposition import RandomizedPCA
import csv
import csvio


train_in = "data/train_inputs.csv"
test_in = "data/test_inputs.csv"

max_rows =None  
max_rows_test = None 

print "Reading Training Data...."
x = csvio.load_csv_data(train_in, max_rows=max_rows)

print "Reading Test Data...."
x_test = csvio.load_csv_data(test_in, max_rows=max_rows_test)

pca = RandomizedPCA(whiten=True)
pca_x= pca.fit_transform(x)
pca_x_test = pca.transform(x_test)

with open("data/pca_train_inputs.csv", "w+") as f: 
  writer = csv.writer(f) 
  writer.writerow(["header"])
  writer.writerows(pca_x)

with open("data/pca_test_inputs.csv", "w+") as f: 
  writer = csv.writer(f) 
  writer.writerow(["header"])
  writer.writerows(pca_x_test)
