import csv
import csvio
from sklearn.preprocessing import scale

train_in = "data/train_inputs.csv"
test_in = "data/test_inputs.csv"

max_rows =None  
max_rows_test = None 

print "Reading Training Data...."
x = csvio.load_csv_data(train_in, max_rows=max_rows)

print "Reading Test Data...."
x_test = csvio.load_csv_data(test_in, max_rows=max_rows_test)

scale_x = scale(x, axis=0, with_mean=True, with_std=True, copy=True)
scale_x_test = scale(x_test, axis=0, with_mean=True, with_std=False, copy=True)

with open("data/scaled_train_inputs.csv", "w+") as f: 
  writer = csv.writer(f) 
  writer.writerow(["header"])
  writer.writerows(scale_x)

with open("data/scaled_test_inputs.csv", "w+") as f: 
  writer = csv.writer(f) 
  writer.writerow(["header"])
  writer.writerows(scale_x_test)
