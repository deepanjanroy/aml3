Preproccessing
All image preprocessing can be done in image_manipulation.py and the functions are documented with intended usage.

Neural Network
To use a neural network, create the network as follows:

import nn
net = nn.NeuralNework(num_input_nodes, num_hidden_layers, [list_of_nodes_per_layer], num_output_nodes, learning_rate)
nn.fit(xs_np_array, ys_np_array)
pred_y = nn.predict(xs_np_array)

If you want to run cross validation, simply run nn.cross_validation().
