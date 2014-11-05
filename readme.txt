Preproccessing
All image preprocessing can be done in image_manipulation.py and the functions are documented with intended usage.

Logistic Regression
The logistic regressor is implemented in logistic/logistic_regression.py as the class MulticlassLogisticRegressor.
It provides fit and predict method similar to scikit learn classifier, with the exception that predict only works
on one example at a time.
 
Neural Network
To use a neural network, create the network as follows:

SVM
We essentially use scikit-learn's implementation of svm, and all testing was done through interactive python sessions.
The final test code for svm can be found in svm.py, if you want to get an idea about how we used svm.

import nn
net = nn.NeuralNework(num_input_nodes, num_hidden_layers, [list_of_nodes_per_layer], num_output_nodes, learning_rate)
nn.fit(xs_np_array, ys_np_array)
pred_y = nn.predict(xs_np_array)

If you want to run cross validation, simply run nn.cross_validation().


