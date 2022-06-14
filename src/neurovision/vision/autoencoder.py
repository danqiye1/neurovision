"""
Autoencoder in Nengo

Author: Ye Danqi
"""
import nengo
import numpy as np
import tensorflow as tf
from nengo_extras.vision import Gabor, Mask

def build_autoencoder(X, n_hidden, n_neurons=1000):
    """
    Build an autoencoder for a images.

    :param X: input data
    :type X: numpy.ndarray (n_samples, H, W)
    :param n_hidden: dimensionality of hidden layer
    :type n_hidden: int
    """
    img_shape = X.shape[1:]
    X_train = X.reshape(len(X), -1)
    size_in = X_train.shape[-1]

    with nengo.Network(label="Autoencoder") as net:
        # Build the neural network
        input = nengo.Node(lambda t: X_train[int(t), :], label="Input")

        # Encoders are used as transforms for dimension reduction
        encoders = Mask(img_shape).populate(
            Gabor().generate(64, (11,11)),
            flatten=True
        )
        # Decoder transforms/weights for reconstruction
        decoders = Mask(img_shape).populate(
            Gabor().generate(64, (11,11)),
            flatten=True
        ).T

        # Ensemble to represent input
        in_ensemble = nengo.Ensemble(
            n_neurons=n_neurons,
            dimensions=size_in
        )

        # Hidden layer
        hidden = nengo.Ensemble(
            n_neurons=500,
            dimensions=n_hidden
        )

        # Ensemble to represent reconstruction
        out_ensemble = nengo.Ensemble(
            n_neurons=n_neurons,
            dimensions=size_in
        )

        nengo.Connection(input, in_ensemble)
        conn1 = nengo.Connection(
            in_ensemble, hidden, 
            transform=encoders,
            learning_rule_type=nengo.PES()
        )
        conn2 = nengo.Connection(
            hidden, out_ensemble,
            transform=decoders,
            learning_rule_type=nengo.PES()
        )
        
        # Ensemble to compute reconstruction error
        recon_error = nengo.Ensemble(n_neurons=n_neurons, dimensions=size_in)

        # Error signal connections
        nengo.Connection(input, recon_error, transform=-1)
        nengo.Connection(out_ensemble, recon_error)
        nengo.Connection(
            recon_error, conn1.learning_rule, 
            function=lambda x: np.sum(np.square(x)) * np.ones(n_hidden)
        )
        nengo.Connection(
            recon_error, conn2.learning_rule, 
            function=lambda x: np.sum(np.square(x)) * np.ones(size_in)
        )

    return net

if __name__=="__main__":
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    net = build_autoencoder(X_train, 64)
    with nengo.Simulator(net) as sim:
        sim.run(20)