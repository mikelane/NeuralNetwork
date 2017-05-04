#!/usr/bin/env python3

"""
Neural Network example for CS445/545 at Portland State University.

This implements a Neural Network (aka, Multi-Level Perceptron) classifier for
the second homework for Machine Learning at Portland State University. I've
tested this on the mnist digit recognition data set, however, it should be able
to handle any discrete (or discretized) data sets. Also, this can (in
principle) handle any number of hidden layers of any size. In practice, this is
limited by time constraints and physical resources, of course.
"""

__author__ = "Mike Lane"
__copyright__ = "Copyright 2017, Michael Lane"
__license__ = "MIT"
__email__ = "mikelane@gmail.com"

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit


class NeuralNetwork:
    """
    The Neural Network class. Minimum initialization requires a number of inputs, number of outputs, as well
    as a list of hidden layer sizes in order of lowest hidden layer to highest hidden layer. The default
    hyperperameters are:

        - Learning Rate = 0.1
        - Momentum = 0.9
        - Random weights seed = [-0.05, 0.05)

    If you've trained a network previously and have the weights as a list of numpy arrays, you can pass
    that in to this network to use it for prediction.

    The Network has statistics built-in including a list of accuracies of the training and test set over
    the training epochs as well as confusion matrices for the epoch with the highest accuracy over the test
    set. Additionally, the weights of the training epoch with the highest test accuracy is stored for later
    use.
    """

    def __init__(self, number_of_inputs: int, number_of_classes: int, hidden_layers_sizes: int,
                 X_train=None, y_train=None, X_test=None, y_test=None,
                 learning_rate=0.1, momentum=0.9, weights=None, rand_low=-0.05, rand_high=0.05):
        """
        NeuralNetwork constructor.
        :param number_of_inputs: integer. Required since a network is targeted to a specific input and output
        :param number_of_classes: integer. Also required for the same reason.
        :param hidden_layers_sizes: integer. Ditto
        :param X_train: The input values of the training set. This must be a numpy array of shape (number
                        of training inputs, number of attributes). This must include a bias value as the
                        0th index. TODO: change this.
        :param y_train: The target values of the training set. This must be a numpy array of shape (number
                        of training inputs, 1). The value of y_train[i] must be the corresponding target
                        class for X_train[i].
        :param X_test: Similar to X_train but for the test set.
        :param y_test: Similar to y_train but for the test set.
        :param learning_rate: float, default 0.1.
        :param momentum: float, default 0.9
        :param weights: List of numpy ndarrays. Default None. If you've already trained this network,
                        you can initialize it with the trained weights
        :param rand_low: The minimum value of a random weight.
        :param rand_high: The maximum value of a random weight.
        """
        # Set the hyperparameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.number_of_classes = number_of_classes
        self.number_of_inputs = number_of_inputs
        self.rand_low = rand_low
        self.rand_high = rand_high
        if weights:  # In case we want to load a pre-trained network
            self.weights = weights
        else:
            # Set the structure of the network. Doing this allows me to make the
            # network have any arbitrary number of layers of any size.
            self.hidden_layers_sizes = hidden_layers_sizes
            self.all_layers_sizes = [number_of_inputs] + hidden_layers_sizes + [number_of_classes]
            self.weight_shapes = [(v + 1, self.all_layers_sizes[i + 1]) for i, v in
                                  enumerate(self.all_layers_sizes[:-1])]
            self.weights = [np.random.uniform(rand_low, rand_high, shape) for shape in self.weight_shapes]
            self.max_accuracy_weights = None
            self.prev_Delta_w = [np.zeros(shape) for shape in self.weight_shapes]
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        # The target vectors for a given class don't change, so set them once and forget them.
        self.targets = np.zeros((number_of_classes, number_of_classes))
        np.fill_diagonal(self.targets, 0.8)
        self.targets += 0.1
        # Initialize the data structures to facilitate analysis.
        self.train_accuracy = []
        self.test_accuracy = []
        self.max_testing_accuracy = 0
        self.max_training_accuracy = 0
        self.train_confusion_matrix = np.zeros((number_of_classes, number_of_classes), dtype=np.int)
        self.test_confusion_matrix = np.zeros((number_of_classes, number_of_classes), dtype=np.int)

    def feed_forward(self, x):
        """
        Forward propagate an instance through the network and return a list of the activations of the hidden
        and output layers. Note: in python 3, the @ is matrix multiply (__matmul__). Wherever you see this
        in the code, it will be the equivalent of using a numpy dot() function.
        :param x: numpy ndarray of the instance attributes with a bias as the 0th index. The shape must be
                  (1, number of attributes)
        :return: List of numpy arrays of shape (1, number of layer values).
        """
        results = []
        if len(self.weights) == 1:  # No hidden layers, so no bias required
            results.append(expit(x @ self.weights[0])) # TODO make sure having no hidden layers works or take this out
        else:  # at least 1 hidden layer
            # Calculate the first hidden layer activations and include the bias.
            results.append(np.insert(expit(x @ self.weights[0]), 0, 1.0, axis=1))
            for w in self.weights[1:-1]:
                # Calculate the rest of the hidden layer activations and include biases
                results.append(np.insert(expit(results[-1] @ w), 0, 1.0, axis=1))
            # Calculate the output layer activations (no bias required).
            results.append(expit(results[-1] @ self.weights[-1]))
        return results

    def error_terms(self, activations, target_vector):
        """
        Calculates the error terms over all hidden layers and the output layer.
        :param activations: A list of activations as numpy ndarrays. Each activation must have the shape
                            (1, size of layer). The activations should be in order of lowest hidden layer
                            to output layer.
        :param target_vector: An ndarray of the shape (1, number of attributes).
        :return: List of Numpy ndarrays of error terms each with shape (layer j, layer k) where j is the
                 layer directly below k.
        """
        # Output error term: $\del_k = o_k * (1 - o_k) (t_k - o_k), \forall k \in o$
        results = [activations[-1] * (1 - activations[-1]) * (target_vector - activations[-1])]

        # Zip together the hidden activations with the weights above them and reverse it
        for i, (a, w) in enumerate(list(zip(activations[:-1], self.weights[1:]))[::-1]):
            # Hidden error term: $\del_j = h_j (1 - h_j) (\Sigma_{k \in o} w_{kj} \del_k)$
            results.append(a[:, 1:] * (1 - a[:, 1:]) * (results[-1] @ w.T[:, 1:]))
        return results[::-1]  # Return the list in reverse to match up with everything else

    def Delta_w(self, del_ws, activations, x):
        """
        Calculate the total value to add to each of the corresponding weights of the corresponding layer.
        :param del_ws: List of ndarrays of error terms. Each should have the same shape as the
                       corresponding weight layer.
        :param activations: List of ndarrays. The activations of each of the hidden layers and the output
                            layer in order. The shape of each must be (1, number of activations of the
                            corresponding layer)
        :param x: Numpy ndarray of input values in the shape of (1, number of attributes)
        :return: List of numpy ndarrays with the same shape as the network's weight values.
        """
        del_ws = [self.learning_rate * d for d in del_ws]
        # Delta_w for first weight matrix
        result = [(x.T @ del_ws[0]) + (self.momentum * self.prev_Delta_w[0])]
        # Delta_w for all subsequent weight matrices
        for i, (a, d, m) in enumerate(list(zip([a.T for a in activations[:-1]],
                                               del_ws[1:],
                                               self.prev_Delta_w[1:]))):
            result.append((a @ d) + (self.momentum * m))
        return result

    def train_network(self):
        """
        Go through the process of forwarding an input to the output layer, calculating the error signal,
        and updating the weights accordingly.
        :return: None
        """
        for x, y in zip(self.X_train, self.y_train):
            x = np.atleast_2d(x)
            activations = self.feed_forward(x)
            Dw = self.prev_Delta_w = self.Delta_w(self.error_terms(activations, self.targets[y]),
                                                  activations, x)
            for i, _ in enumerate(self.weights):
                self.weights[i] += Dw[i]

    def predict(self, x):
        """
        Similar process of feed_forward, but disregard intermediate hidden layer values. For a network with
        a single hidden layer, this could, conceivably be a one-line function. But with an unknown number of
        layers (and to avoid the stack growing too much with recursion), I separated it out this way.
        :param x: Numpy ndarray. The instance attribute values in shape (1, number of attributes)
        :return: int. The class that this network guesses for the given input value.
        """
        result = None
        if len(self.weights) == 1:  # No hidden layers, so no bias required
            result = expit(x @ self.weights[0])
        else:  # at least 1 hidden layer
            # Calculate the first hidden layer activations and include the bias.
            result = np.insert(expit(x @ self.weights[0]), 0, 1.0, axis=1)
            for w in self.weights[1:-1]:
                # Calculate the rest of the hidden layer activations and include biases
                result = np.insert(expit(result @ w), 0, 1.0, axis=1)
            # Calculate the output layer activations (no bias required).
            result = expit(result @ self.weights[-1])
        return np.argmax(result)

    def calculate_accuracy(self):
        """
        For each input predict the output and tally the number of correct guesses. Also, update the confusion
        matrix at each step.
        :return: None.
        """
        cm_train = np.zeros_like(self.train_confusion_matrix, dtype=np.int)
        cm_test = np.zeros_like(self.test_confusion_matrix, dtype=np.int)

        num_correct = 0
        for i, x in enumerate(self.X_train):
            actual = y_train[i]
            predicted = self.predict(np.atleast_2d(x))
            if actual == predicted:
                num_correct += 1
            cm_train[actual][predicted] += 1
        self.train_accuracy.append(num_correct / len(self.X_train))
        if self.train_accuracy[-1] > self.max_training_accuracy:
            self.max_training_accuracy = self.train_accuracy[-1]
            self.train_confusion_matrix = cm_train

        num_correct = 0
        for i, x in enumerate(self.X_test):
            actual = y_test[i]
            predicted = self.predict(np.atleast_2d(x))
            if actual == predicted:
                num_correct += 1
            cm_test[actual][predicted] += 1
        self.test_accuracy.append(num_correct / len(self.X_test))
        if self.test_accuracy[-1] > self.max_testing_accuracy:
            self.max_accuracy_weights = self.weights.copy()
            self.max_testing_accuracy = self.test_accuracy[-1]
            self.test_confusion_matrix = cm_test

    def fit(self, number_of_epochs=50, X_train=None, y_train=None, X_test=None, y_test=None, verbose=False,
            momentum=None):
        """
        Train the network weights for a given set of inputs.
        :param number_of_epochs: int, default 50.
        :param X_train: The training set to use. Required if network wasn't constructed with it.
        :param y_train: The testing set to use. Required if network wasn't constructed with it.
        :param X_test: Same
        :param y_test: Same
        :param verbose: bool, default False. To get fancy pants visual feedback on the training progress.
        :param momentum: float, default None. Allow each test to modify the momentum value.
        :return: None
        """
        from datetime import datetime
        # Allow overriding test sets
        self.X_train = X_train if X_train else self.X_train
        self.y_train = y_train if y_train else self.y_train
        self.X_test = X_test if X_test else self.X_test
        self.y_test = y_test if y_test else self.y_test
        self.momentum = momentum if momentum != None else self.momentum
        # Train using a fresh set of random weights.
        self.weights = [np.random.uniform(self.rand_low, self.rand_high, shape) for shape in
                        self.weight_shapes]
        self.train_accuracy = []
        self.test_accuracy = []

        if verbose:
            print('\r|>{}| Dur: {} (Avg: {:.2f}s/epoch) Est. Completion Time: ?'.format(
                ' ' * number_of_epochs,
                '0:00:00.000000',
                0.0),
                end='',
                flush=True)
        start = datetime.now()

        for epoch in range(number_of_epochs):
            # TODO add in shuffling of the input data
            self.train_network()
            self.calculate_accuracy()
            if verbose:
                print('\r|{}>{}| '
                      'Dur: {} (Avg {:.2f}s/epoch) '
                      'Est. Completion Time: {}'.format(
                    '=' * (epoch + 1), ' ' * (number_of_epochs - (epoch + 1)),
                    datetime.now() - start,
                    ((datetime.now() - start) / (epoch + 1)).total_seconds(),
                    (datetime.now() + (datetime.now() - start) / (epoch + 1) * (
                        number_of_epochs - (epoch + 1)))
                ),
                    end='\n' if epoch + 1 == number_of_epochs else '',
                    flush=True)

        # Use matplotlib to plot data and save it to disk.
        print('Plotting Data')
        plt.figure()
        self.plot_data('Accuracies_eta{}_alpha{}_hidden{}.png'.format(self.learning_rate,
                                                                      self.momentum,
                                                                      self.hidden_layers_sizes))
        plt.figure()
        self.plot_cm(self.train_confusion_matrix,
                     r'Training $\eta$:{} $\alpha$:{} hidden:{}'.format(self.learning_rate,
                                                                        self.momentum,
                                                                        self.hidden_layers_sizes),
                     'Confusion_Matrix_eta{}_alpha{}_hidden{}.png'.format(self.learning_rate,
                                                                          self.momentum,
                                                                          self.hidden_layers_sizes))
        plt.figure()
        self.plot_cm(self.test_confusion_matrix,
                     r'Testing $\eta$:{} $\alpha$:{} hidden:{}'.format(self.learning_rate,
                                                                       self.momentum,
                                                                       self.hidden_layers_sizes),
                     'Testing_Confusion_Matrix_eta{}_alpha{}_hidden{}.png'.format(self.learning_rate,
                                                                                  self.momentum,
                                                                                  self.hidden_layers_sizes))

    def plot_data(self, filename):
        """
        Plot the accuracy data using matplotlib. For more information, look at the docs.
        :param filename: The filename to save this figure.
        :return: None
        """
        fig, ax = plt.subplots(figsize=(10, 7.5))
        fig.suptitle('Neural Network Accuracy\n'
                     r'$\eta$: {} $\alpha$: {} Hidden Layers: {}'.format(self.learning_rate,
                                                                         self.momentum,
                                                                         self.hidden_layers_sizes),
                     fontsize=18,
                     fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.tick_params(axis='x')
        ax.tick_params(axis='y')
        ax.yaxis.grid(True)
        line1, = ax.plot(range(len(self.train_accuracy)),
                         self.train_accuracy,
                         label='training',
                         linewidth=2)
        line2, = ax.plot(range(len(self.test_accuracy)),
                         self.test_accuracy,
                         label='testing',
                         linewidth=2)
        ax.legend(loc='lower right')
        plt.savefig(filename, dpi=300)

    def plot_cm(self, cm, title, filename):
        """
        Plot a given confusion matrix. See matplotlib for more information.
        :param cm: Numpy ndarray. The confusion matrix to plot
        :param title: str. The title to use.
        :param filename: str. the filename
        :return: None
        """
        import itertools
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, aspect='equal')
        plt.title(title, fontsize=16, fontweight='bold')
        cb = plt.colorbar()
        cbytick_obj = plt.getp(cb.ax.axes, 'yticklabels')
        # plt.setp(cbytick_obj)
        tick_marks = range(10)
        plt.xticks(tick_marks, range(self.number_of_classes))
        plt.yticks(tick_marks, range(self.number_of_classes))
        plt.ylabel('Actual', fontsize=14)
        plt.xlabel('Predicted', fontsize=14)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=6)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)


if __name__ == '__main__':
    def load_data(filename):
        """
        Load the mnist digit data.
        :param filename: The filename of the data.
        :return: Tuple of ndarrays corresponding to (target values, instances)
        """
        import pandas as pd
        from datetime import datetime
        start = datetime.now()
        print('Loading {}'.format(filename), end=' ... ', flush=True)
        data = pd.read_csv(filename, header=None, index_col=0) / 255.0
        # TODO Add this bias in the code.
        data.insert(0, 'bias', np.ones((data.shape[0], 1)))
        print('DONE {}s'.format((datetime.now() - start).total_seconds()), flush=True)
        return data.values, data.index.values


    # Load the data
    X_train, y_train = load_data('https://pjreddie.com/media/files/mnist_train.csv')
    X_test, y_test = load_data('https://pjreddie.com/media/files/mnist_test.csv')

    nn20 = NeuralNetwork(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                         number_of_inputs=784, number_of_classes=10, hidden_layers_sizes=[20])
    nn50 = NeuralNetwork(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                         number_of_inputs=784, number_of_classes=10, hidden_layers_sizes=[50])
    nn100 = NeuralNetwork(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                          number_of_inputs=784, number_of_classes=10, hidden_layers_sizes=[100])
    # nn20.fit(number_of_epochs=5, verbose=True)
    # Run the following tests
    nn20.fit(number_of_epochs=50, verbose=True)
    nn50.fit(number_of_epochs=50, verbose=True)
    nn100.fit(number_of_epochs=50, verbose=True)
    nn100.fit(number_of_epochs=50, verbose=True, momentum=0)
    nn100.fit(number_of_epochs=50, verbose=True, momentum=0.25)
    nn100.fit(number_of_epochs=50, verbose=True, momentum=0.5)
    # I didn't take the time to modify the testing and training data sets as per the assignment.
    print('END')
