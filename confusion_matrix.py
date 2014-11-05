from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import csv 

vf = "valid_pred.csv"
vf1 = "valid_pred_actual.csv"

y_pred =[] 
y_actual = []

with open(vf, "r") as f: 
  reader = csv.reader(f)
  next(reader) 
  for row in reader: 
    y_pred.append(int(row[1]))


with open(vf1, "r") as f: 
  reader = csv.reader(f)
  next(reader) 
  for row in reader: 
    y_actual.append(int(row[1]))
# Compute confusion matrix
cm = confusion_matrix(y_actual, y_pred)
print(cm)

# Show confusion matrix in a separate window
plt.matshow(cm)
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
