"""
Convolutional Neural Network for building a Visual Hierarchy.

Adapted from https://www.nengo.ai/nengo-dl/examples/spiking-mnist.html.
This can be used as a blueprint for building future visual hierarchy models.
"""
import nengo
import nengo_dl
import numpy as np
import tensorflow as tf

def build_visual_hierarchy(shape_in, size_out):
    """
    Function for constructing a nengo network.

    :param shape_in: dimension of input (usually is number of image pixels)
    :param size_out: dimension of output (usually is number of classes)
    """
    with nengo.Network(seed=0) as net:
        # Default parameters for all neurons
        net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
        net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
        net.config[nengo.Connection].synapse = None

        # Optimisation for improving training speed
        nengo_dl.configure_settings(stateful=False)

        # Build the neural network
        input = nengo.Node(np.zeros(np.prod(shape_in)))
        conv1 = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=32, kernel_size=3))
        conv2 = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=64, strides=2, kernel_size=3))
        conv3 = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=128, strides=2, kernel_size=3))
        ensemble_layer = nengo_dl.Layer(nengo.LIF(amplitude=0.01))
        out_layer = nengo_dl.Layer(tf.keras.layers.Dense(units=size_out))

        # Forward pass
        x = conv1(input, shape_in=shape_in)
        x = ensemble_layer(x)
        x = conv2(x, shape_in=(26, 26, 32))
        x = ensemble_layer(x)
        x = conv3(x, shape_in=(12, 12, 64))
        x = ensemble_layer(x)
        out = out_layer(x)

    return net

def train_network(sim, X_train, y_train, epochs):
    sim.compile(
        optimizer=tf.optimizers.RMSprop(0.001),
        loss={"out_p": tf.losses.SparseCategoricalCrossentropy(from_logits=True)}
    )

    sim.fit(X_train, {"out_p": y_train}, epochs=epochs)
    return sim
    

if __name__ == "__main__":
    # Load and preprocess data
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    shape = X_train.shape
    X_train = X_train.reshape((len(X_train), -1))
    X_test = X_test.reshape((len(X_test), -1))
    # Add time dimension
    X_train = X_train[:, None, :]
    y_train = y_train[:, None, None]
    # Repeat input/target for n_steps
    n_steps = 30
    X_test = np.tile(X_test[:, None, :], (1, n_steps, 1))
    y_test = np.tile(y_test[:, None, None], (1, n_steps, 1))

    # Build network and simulator
    net = build_visual_hierarchy((28,28,1), 10)
    minibatch_size = 200
    sim = nengo_dl.Simulator(net, minibatch_size=minibatch_size)
    train_network(sim, X_train, y_train, 10)
