# Generating some random data

import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from numpy.linalg import norm

from exceptions import OverflowError


class LogisticRegressor(object):
    # TODO: Normalize features
    def __init__(self):
        self.weights = None
        self.S = lambda x: 1.0 / ( 1 + math.exp(-x))  # Sigmoid function
        self.threshold = 0.5
        self.learning_rate = .001
        self.gd_stop_crit = 0.0001
        self.C = 0.1  # Please use a float

        # For momentum in gradient_descent
        self.momentum = 0.9
        self.previous_update = np.zeros_like(self.weights)

    def predict_proba(self, x):
        proper_input = np.concatenate(([1], x))
        inner_prod = np.inner(self.weights, proper_input)
        return self.S(inner_prod)


    def predict(self, x):
        if self.predict_proba(x) > self.threshold:
            return 1
        else:
            return 0

    def pointwise_error(self, x, y):
        inner_prod = np.inner(self.weights, x)
        return math.log(1 + math.exp(-y * inner_prod))

    def bind_data(self, X, y):
        ones = np.array([[1]] * len(X))
        self.X = np.concatenate((ones, X), axis=1)
        self.y = y


    def fit(self, X, y, max_gd_steps=10000, random_restarts=5, sgd_batch_size=10):
        self.bind_data(X, y)

        min_weights = None
        min_error = np.inf

        for run in xrange(random_restarts):
            self.weights = np.zeros(len(self.X[0]))
            self.previous_update = 0

            # We want to get a non-zero previous update
            max_tries = 1000
            try_num = 0
            while norm(self.previous_update) == 0 and try_num < max_tries:
                self.one_step_sgd(sgd_batch_size)
                try_num += 1

            if norm(self.previous_update) == 0:
                print "Could not perform gradient descent with the current initial weights"
                print "Giving up on the current run"
                continue

            for i in xrange(max_gd_steps):
                # Uncomment this for early break
                # if norm(self.previous_update) < self.gd_stop_crit:
                #     print "Hello for run {0}: Last update was {1}".format(run, norm(self.previous_update))
                #     print "Not trying to descend further."
                #     break

                self.one_step_sgd(sgd_batch_size)

            if self.train_error() < min_error:
                min_error = self.train_error()
                min_weights = self.weights.copy()

        self.weights = min_weights

    def train_error(self, regularized=True):
        total = 0
        for i in xrange(len(self.X)):
            total += self.pointwise_error(self.X[i], self.y[i])

        w_without_bias = self.weights[1:]
        w_sq_sum = w_without_bias.T.dot(w_without_bias)

        if regularized:
            regterm = self.C / 2 * w_sq_sum
        else:
            regterm = 0
        return float(total) / len(self.X) + regterm

    def gradient(self, w):
        """
        TODO(Refactor): get rid of the w in params
        """
        total = np.zeros(len(self.weights))
        for i in xrange(len(self.X)):
            inner_prod = np.inner(np.array(w), self.X[i])
            denom = 1 + math.exp(self.y[i] * inner_prod)
            total += self.y[i] * self.X[i] / denom

        w = self.weights.copy()
        w[0] = 0
        return -1.0/len(self.X) * total + self.C * w

    def stochastic_grad(self, w, batch_size=10):
        indices = np.random.random_integers(0, len(self.X) - 1, batch_size)
        total = 0
        for i in indices:
            inner_prod = np.inner(np.array(w), self.X[i])
            denom = 1 + math.exp(self.y[i] * inner_prod)
            total += self.y[i] * self.X[i] / denom

        w = self.weights.copy()
        w[0] = 0
        return -1.0 / len(indices) * total + self.C * w

    def one_step_gd(self):
        try:
            update = self.learning_rate * self.gradient(self.weights) \
                    + self.momentum * self.previous_update
            self.weights -= update
            self.previous_update = update

        except OverflowError:
            print "OverflowError, setting weights to zero"
            self.weights = np.zeros(len(self.weights))
            self.previous_update = 0

    def one_step_sgd(self, batch_size=10):
        try:
            update = self.learning_rate * self.stochastic_grad(self.weights) \
                    + self.momentum * self.previous_update
            self.weights -= update
            self.previous_update = update
        except OverflowError:
            print "OverflowError, not updating weights"


class MulticlassLogisticRegressor(object):
    """
    Uses n one-vs-all binary logistic regressors
    """

    def __init__(self):
        pass

    def bind_data(self, X, y):
        self.classes = list(set(y))
        self.class_dict = { self.classes[i]: i for i in xrange(len(self.classes)) }
        self.y = y
        self.X = X


    def fit(self, X, y, max_gd_steps=10000, random_restarts=5, sgd_batch_size=10):
        self.bind_data(X, y)
        self.binary_regressors = []


        for kls in self.classes:
            y_for_kls = []
            for elm in self.y:
                if elm == kls:
                    y_for_kls.append(1)
                else:
                    y_for_kls.append(-1)

            logreg_for_kls = LogisticRegressor()
            logreg_for_kls.fit(self.X, y_for_kls,
                               max_gd_steps=max_gd_steps,
                               random_restarts=random_restarts,
                               sgd_batch_size=sgd_batch_size)

            self.binary_regressors.append(logreg_for_kls)

    def predict_proba(self, x, class_ind):
        bin_reg = self.binary_regressors[class_ind]
        return bin_reg.predict_proba(x)

    def predict(self, x):
        max_class_ind = 0
        max_score = 0

        for i in xrange(len(self.classes)):
            prob = self.predict_proba(x, i)
            if prob > max_score:
                max_score = prob
                max_class_ind = i

        return self.classes[max_class_ind]

    def train_accuracy(self):
        total_correct = 0
        for i in xrange(len(self.X)):
            if self.predict(self.X[i]) == self.y[i]:
                total_correct += 1

        return float(total_correct) / len(self.X)

    def score(self, X, y):
        total_correct = 0
        for i in xrange(len(X)):
            if self.predict(X[i]) == y[i]:
                total_correct += 1
        return float(total_correct) / len(X)



if (len(sys.argv) > 1 and sys.argv[1] == "test" ):
    """
    Code for interactively testing the implementation

    You can use `ipython -i logistic_reg.py test`, play
    around with some of these things
    """

    glbs = {}  # Global variable container

    # Generate random x and y coordinates in the range (0,100)

    x = np.random.ranf(100) * 100
    y = np.random.ranf(100) * 100

    # Defining the line
    line_eq = lambda x: 0.5 * x + 10
    line_x = np.array([0, 100])

    # y = .5x + 10
    line_y = line_eq(line_x)

    # Make data more clearly separable
    tolerance = 2
    for i in xrange(len(x)):
        # If close to line, shift it in y direction
        if abs(y[i] - line_eq(x[i])) < tolerance:
            if y[i] < 50:
                y[i] += 5
            else:
                y[i] -= 5

    X = np.array(zip(x, y))
    Y = np.zeros(len(X))

    for i in xrange(len(X)):
        point = X[i]
        if line_eq(point[0]) > point[1]:
            Y[i] = 1
        else:
            Y[i] = -1

    for datam in zip(X, Y):
        if datam[1] == 1:
            color='g'
        else:
            color='r'
        plt.plot(*datam[0], color=color, linestyle=" ", marker="o")

    # plt.plot(x, y, marker="o", linestyle=" ")
    plt.plot(line_x, line_y)
    plt.show(block=False)


    def plot_db(plt, weights=(0,0,0)):
        """
        Plots decision boundary given weights. Weights should be a 3-tuple (w_0, w_1, w_2).
        """

        if "old_decision_boundary" in glbs:
            for shape in glbs["old_decision_boundary"]:
                shape.remove()
            del glbs["old_decision_boundary"]

        w_0, w_1, w_2 = weights

        if w_1 == 0 and w_2 == 0:
            print "Warning: No decision boundary can be plotted - both x_1 and x_2 have zero weight."
            return
        elif w_2 == 0:
            x = - float(w_0) / w_1
            line_x = [x, x]
            line_y = [0, 100]
        else:
            line_x = [0, 100]
            line_eq = lambda x: - 1.0 / w_2 * (w_0 + w_1 * x)
            line_y = [line_eq(x) for x in line_x]

        shapes = plt.plot(line_x, line_y, color="orange")
        ax = plt.gca()
        ax.relim()
        ax.autoscale_view()

        # Shade the positive half
        ax.get_axes().set_autoscale_on(False)

        # Check which half (0,100) belongs to
        if w_0 + w_2 * 100 > 0:
            polyshape = plt.fill_between(line_x, 100, line_y, color="green", alpha=0.4)
        else:
            polyshape = plt.fill_between(line_x, line_y, 0, color="green", alpha=0.4)

        shapes.append(polyshape)
        glbs["old_decision_boundary"] = shapes
        plt.draw()
        ax.get_axes().set_autoscale_on(True)

    # debug
    plot_db(plt, (0, 5, -10))

    def animate(plt):
        "Test code to see if I can animate the decision boundary in the plot"
        yr = range(41)
        yr = np.array(yr, dtype=float)
        yr /= 10
        yr -= 5
        for yd in yr:
            plot_db(plt, (0, 5, yd))
            plt.pause(1)

    logreg = LogisticRegressor()
