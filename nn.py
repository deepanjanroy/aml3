import numpy as np

class NeuralNetwork(object):
	class Layer(object):
		def __init__(self, n_nodes, n_inputs, b_input, b_output):
			''' Creates the weight matrices, node values, and error matrices needed for the current node. 
			@param n_nodes The number of nodes to be included in the current layer.
			@param n_inputs The number of nodes feeding into this layer (0 for input layer).
			@param b_input True if this layer represents the input layer.
			@param b_output True if this layer is the output layer.
			'''
			if not b_input:
				self.weights = np.zeros((n_nodes,n_inputs+1)) # Note the +1 is for the bias node.
				self.delta = np.zeros((n_nodes,1)) 
			if not b_output:
				self.xs = np.zeros((n_nodes+1,1)) # The output layer does not have a bias node.
			else:
				self.xs = np.zeros((n_nodes,1))

			
	def __init__(self, n_input, n_hidden, n_nodes_per_layer, n_output):
		''' Takes information about constructing the neural network.
		@param n_input The number of nodes to include in the input layer.
		@param n_hidden The number of hidden layers to be included in the network.
		@param n_nodes_per_layer A list of length n_hidden of the number of nodes in each hidden layer.
		@param n_output The number of nodes to be included in the output layet.
		'''
		self.n_outputs = n_output
		self.input_layer = NeuralNetwork.Layer(n_input, 0, True, False)
		self.hidden_layers = []
		for i in xrange(0, len(n_nodes_per_layer)):
			if i == 0:
				self.hidden_layers.append(NeuralNetwork.Layer(n_nodes_per_layer[i], n_input, False, False))
			else:
				self.hidden_layers.append(NeuralNetwork.Layer(n_nodes_per_layer[i], n_nodes_per_layer[i-1], False, False))
		self.output_layer = NeuralNetwork.Layer(n_output, n_nodes_per_layer[-1], False, True)
		
		# Create the sigmoid function to be used by nodes.
		def f(x): return 1/(1+np.exp(-x))
		self.sigmoid = np.vectorize(f)
		
	def forwardProp(self, input_xs):
		''' Given an input layer, propogate the values through the network to the output nodes.
		@param input_xs The vector of inputs to the network.
		'''
		self.input_layer.xs[0] = 1 # Add a bias node.
		self.input_layer.xs[1:,0] = input_xs
		for i in xrange(0, len(self.hidden_layers)):
			h = self.hidden_layers[i]
			h.xs[0] = 1 # Add a bias node.
			if i == 0:
				h.xs[1:] = self.sigmoid(np.dot(h.weights, self.input_layer.xs))
			else:
				h.xs[1:] = self.sigmoid(np.dot(h.weights, self.hidden_layers[i-1].xs))
		self.output_layer.xs = self.sigmoid(np.dot(self.output_layer.weights, self.hidden_layers[-1].xs))
		
	def backProp(self, alpha, y):
		''' Given an alpha value to use in gradient descent, calculate new weights based on the error of the output. 
		@param alpha Learning rate for gradient descent.
		@param y The correct output layer values.
		'''
		temp = np.multiply(self.output_layer.xs, np.subtract(np.ones(self.output_layer.xs.shape), self.output_layer.xs))
		self.output_layer.delta = np.multiply(temp, np.subtract(y,self.output_layer.xs))
		#Propogate the error.
		for i in xrange(len(self.hidden_layers)-1, -1, -1):
			h = self.hidden_layers[i]
			if i == 0:
				prev = self.input_layer
			else:
				prev = self.hidden_layers[i-1]
			if i == len(self.hidden_layers)-1:
				next = self.output_layer
				temp1 = np.dot(next.weights.T, next.delta)
			else:
				next = self.hidden_layers[i+1]
				temp1 = np.dot(next.weights.T, next.delta[1:])
			temp2 = np.multiply(h.xs, np.subtract(np.ones(h.xs.shape), h.xs))
			h.delta = np.multiply(temp1, temp2)
		# Update the weights.
		for i in xrange(len(self.hidden_layers)-1, -1, -1):
			h = self.hidden_layers[i]
			if i == 0:
				prev = self.input_layer
			else:
				prev = self.hidden_layers[i-1]
			h.weights += alpha*np.dot(h.delta[1:], prev.xs.T)	
		self.output_layer.weights += alpha * np.dot(self.output_layer.delta, self.hidden_layers[-1].xs.T)	
		
	def fit(self, xs_train, ys_train, eps=0.0001, max_iter=100, alpha=1):
		''' Given a set of inputs and outputs, perform stochastic graidient descent until we have converged.
		@param xs_train A matrix of training examples.
		@param ys_train A vector of outputs for the training examples.
		@param max_iter The maximum number of iterations of stochastic gradient descent to perform.
		@param alpha The learning rate for the backProp algorithm.
		'''
		# Initialize the nodes with random weights.
		for h in self.hidden_layers:
			h.weights = np.random.randint(0,1000,h.weights.shape)*0.001
		self.output_layer.weights = np.random.randint(0,1000,self.output_layer.weights.shape)*0.001
		# Perform one-hot encoding on the output class if needed.s
		if self.n_outputs != 1:	
			ys = np.zeros((ys_train.shape[0],self.n_outputs))
			for i in xrange(0, ys_train.shape[0]):
				ys[i, ys_train[i]] = 1
		else:
			ys = ys_train
		
		# Keep track of how many outputs in a row have been less than a given value.
		converge = 0
		prev = 0
		for _ in xrange(0, max_iter):
			for i in xrange(0, xs_train.shape[0]):
				self.forwardProp(xs_train[i,:])
				if self.n_outputs != 1:
					self.backProp(alpha, np.reshape(ys[i], (self.n_outputs,1)))
				else:	
					self.backProp(alpha, np.array([ys[i]]))
			if abs(prev - self.output_layer.xs[0]) < eps:
				converge+=1
			else:
				converge = 0
			prev = self.output_layer.xs[0]
			if converge > 10000:
				break
				
	def predict(self, xs_test):
		''' Given a single training example, return a prediction for its class.
		@param xs_test A single training example.
		'''
		self.forwardProp(xs_test)
		if self.n_outputs == 1:
			if self.output_layer.xs[0] > 0.5:
				return 1
			else:
				return 0
		return np.argmax(self.output_layer.xs)
		
if __name__ == '__main__':
	import csv
	# Test the XOR function.
	xor_list = [[0, 0], [0, 1], [1, 0], [1, 1]]
	ys_list = [0, 1, 1, 0]
	nn = NeuralNetwork(2304, 1, [2], 10)
	xs_train = np.load('../data_and_scripts/train_inputs.npy')
	ys1 = []
	with open('../data_and_scripts/train_outputs.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		next(reader, None)  # skip the header
		for train_output in reader:  
		    train_output_no_id = int(train_output[1])
		    ys1.append(train_output_no_id)
	ys1 = np.array(ys1)
	print "Training"
	nn.fit(xs_train[0:1000],ys1[0:1000])
	for i in xrange(1000, 2000):
		pred = nn.predict(xs_train[i])
		print pred == ys1[i]
	
