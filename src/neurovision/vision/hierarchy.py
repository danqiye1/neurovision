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
        out_layer = nengo_dl.Layer(tf.keras.layers.Dense(units=10))

        # Forward pass
        x = conv1(input, shape_in=shape_in)
        x = ensemble_layer(x)
        x = conv2(x, shape_in=(26, 26, 32))
        x = ensemble_layer(x)
        x = conv3(x, shape_in=(12, 12, 64))
        x = ensemble_layer(x)
        out = out_layer(x)

    return net
