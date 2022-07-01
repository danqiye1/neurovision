"""
An autoencoder experiment to see if we can do localised learning.

Resolved:
1. Dimensions for transform is ok

Current problems:
1. Credit assignment of reconstruction loss.
2. Unable to debug on UI since rendering vector of 784 is laggy
"""

import nengo
import numpy as np
import tensorflow as tf
from nengo_extras.gui import image_display_function
from nengo_extras.vision import Gabor, Mask

model = nengo.Network()

rng = np.random.RandomState(9)

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
img_dim = X_train[0].reshape(-1).shape[0]
hidden_dim = 392

def preprocess(img):
    X = img.reshape(-1)
    X = X - np.mean(X)
    return X / np.linalg.norm(X)

with model:
    model.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
    model.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    model.config[nengo.Connection].synapse = None
    
    input = nengo.Node(lambda t: preprocess(X_train[int(t) // 2]))
    
    # Encoders are used as transforms for dimension reduction
    encoders = Mask((28,28)).populate(
        Gabor().generate(hidden_dim, (11,11), rng=rng),
        rng=rng, 
        flatten=True
    )
    # Decoder transforms/weights for reconstruction
    decoders = Mask((28, 28)).populate(
        Gabor().generate(hidden_dim, (11,11), rng=rng),
        rng=rng,
        flatten=True
    )
    
    in_ensemble = nengo.networks.EnsembleArray(
        n_neurons=30,
        n_ensembles=img_dim,
        ens_dimensions=1
    )

    hidden = nengo.Ensemble(
        n_neurons=5000,
        dimensions=hidden_dim
    )

    out_ensemble = nengo.networks.EnsembleArray(
        n_neurons=30,
        n_ensembles=img_dim,
        ens_dimensions=1
    )
    
    # Feedforward connections
    nengo.Connection(input, in_ensemble.input)
    conn1 = nengo.Connection(
        in_ensemble.output, hidden, 
        transform=encoders
    )
    conn2 = nengo.Connection(
        hidden, out_ensemble.input, 
        transform=decoders.T,
        learning_rule_type=nengo.PES()
    )
    
    # Error signal calculation
    # and error signal connection
    recon_error = nengo.Ensemble(
        n_neurons=784,
        dimensions=784
    )
    nengo.Connection(out_ensemble.output, recon_error, transform=-1)
    nengo.Connection(in_ensemble.output, recon_error)
    nengo.Connection(
        recon_error, conn2.learning_rule
    )

    # Input image display (for nengo_gui)
    image_shape = (1, 28, 28)
    display_func = image_display_function(image_shape, offset=1, scale=128)
    display_node = nengo.Node(display_func, size_in=input.size_out)
    nengo.Connection(input, display_node, synapse=None)

    output = nengo.Node(display_func, size_in=img_dim)
    nengo.Connection(out_ensemble.output, output, synapse=0.1)
    
    
