"""
Autoencoder in Nengo

Author: Ye Danqi
"""
import nengo
import numpy as np
import tensorflow as tf

from pdb import set_trace as bp

def build_autoencoder(X_train, n_hidden):
    with nengo.Network(label="Autoencoder") as net:
        # Default parameters for all neurons
        net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
        net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
        net.config[nengo.Connection].synapse = None

        # Build the neural network
        size_in = X_train.shape[-1]
        input = nengo.Node(np.zeros(size_in), label="Input")

        # Hidden layer for dimension reduction
        hidden = nengo.Ensemble(
            n_neurons=500,
            dimensions=n_hidden
        )
        nengo.Connection(input, hidden)

        # Output layer for reconstruction
        out_layer = nengo.Ensemble(
            n_neurons=1000,
            dimensions=size_in
        )
        conn = nengo.Connection(hidden, out_layer, eval_points=X_train, function=X_train)
        conn.learning_rule_type = nengo.PES()
        
        # Ensemble to compute reconstruction error
        recon_error = nengo.Ensemble(100, dimensions=1)
        nengo.Connection(input, recon_error, transform=-1)
        nengo.Connection(out_layer, recon_error)
        nengo.Connection(recon_error, conn.learning_rule, function=lambda x: np.sum(np.square(x)))

    return net

if __name__=="__main__":
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    net = build_autoencoder(X_train.reshape(len(X_train), -1), 64)
    bp()