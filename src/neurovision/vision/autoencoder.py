"""
Autoencoder in Nengo

Author: Ye Danqi
"""
import nengo
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
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
            Gabor().generate(n_hidden, (11,11)),
            flatten=True
        )
        # Decoder transforms/weights for reconstruction
        decoders = Mask(img_shape).populate(
            Gabor().generate(n_hidden, (11,11)),
            flatten=True
        ).T

        # Ensemble to represent input
        in_ensemble = nengo.Ensemble(
            n_neurons=n_neurons,
            dimensions=size_in
        )

        # Hidden layer
        hidden = nengo.Ensemble(
            n_neurons=n_neurons,
            dimensions=n_hidden
        )

        # Ensemble to represent reconstruction
        out_ensemble = nengo.Ensemble(
            n_neurons=n_neurons,
            dimensions=size_in,
            label="Output"
        )

        nengo.Connection(input, in_ensemble)
        conn1 = nengo.Connection(
            in_ensemble, hidden, 
            transform=encoders,
            learning_rule_type=nengo.PES(learning_rate=0.001)
        )
        conn2 = nengo.Connection(
            hidden, out_ensemble,
            transform=decoders,
            learning_rule_type=nengo.PES(learning_rate=0.001)
        )
        
        # Ensemble to compute reconstruction error
        recon_error = nengo.Ensemble(
            n_neurons=n_neurons, 
            dimensions=size_in,
            label="Error"
        )

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
    
    # Add some probes
    input = net.nodes[0]
    output = net.ensembles[-2]
    error = net.ensembles[-1]
    with net:
        in_probe = nengo.Probe(input, synapse=0.01)
        out_probe = nengo.Probe(output, synapse=0.01)
        err_probe = nengo.Probe(error, synapse=0.01)

    with nengo.Simulator(net) as sim:
        sim.run(20)

    plt.figure()
    plt.plot(sim.trange(), sim.data[in_probe])
    plt.savefig("input.png")
    plt.close()

    plt.figure()
    plt.plot(sim.trange(), sim.data[out_probe])
    plt.savefig("output.png")
    plt.close()

    plt.figure()
    plt.plot(sim.trange(), sim.data[err_probe])
    plt.savefig("error.png")
    plt.close()