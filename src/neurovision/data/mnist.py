"""
MNIST dataset class for use with Nengo
"""
import tensorflow as tf
from nengo_extras.data import one_hot_from_labels

class MNIST(object):
    """
    MNIST dataset for loading and generating data for Nengo.
    """

    def __init__(self, path="mnist.npz", training=True):
        """ Load or download dataset from/to path """

        # Load data
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data(path=path)
        self.X_train = X_train.reshape(len(X_train), -1)
        self.y_train = one_hot_from_labels(y_train)
        self.X_test = X_test.reshape(len(X_test), -1)
        self.y_test = one_hot_from_labels(y_test)

        # Set training/testing mode
        self.training = training

    def __getitem__(self, idx):
        if self.training:
            return (self.X_train[idx], self.y_train[idx])
        else:
            return (self.X_test[idx], self.y_test[idx])

    def __len__(self):
        return len(self.X_train) if self.training else len(self.X_test)

    def dim(self):
        """ Get dimensions """
        return self.X_train.shape[1]

    def num_classes(self):
        """ Get the number of classes """
        return self.y_train.shape[1]

    def get_eval_points(self):
        """ Get the evaluation points for nengo.Ensemble """
        return self.X_train

    def targets(self):
        """ Get the one-hot-encoded targets """
        return self.y_train

    def train(self):
        """ Set training mode """
        self.training = True

    def test(self):
        """ Set testing mode """
        self.training = False
    

