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
X_train = X_train.reshape(len(X_train), -1)
n_neurons = 1000
hidden_dim = 392

with model:
    model.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
    model.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    model.config[nengo.Connection].synapse = None
    
    input = nengo.Node(lambda t: X_train[int(t), :])
    
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
    
    in_ensemble = nengo.Ensemble(
        n_neurons=n_neurons,
        dimensions=X_train.shape[1]
    )

    hidden = nengo.Ensemble(
        n_neurons=n_neurons,
        dimensions=hidden_dim
    )

    out_ensemble = nengo.Ensemble(
        n_neurons=n_neurons,
        dimensions=X_train.shape[1]
    )
    
    # Feedforward connections
    nengo.Connection(input, in_ensemble)
    conn1 = nengo.Connection(
        in_ensemble, hidden, 
        transform=encoders,
    )
    conn2 = nengo.Connection(
        hidden, out_ensemble, 
        transform=decoders.T,
        learning_rule_type=nengo.PES()
    )
    
    # Error signal calculation
    # and error signal connection
    recon_error = nengo.Ensemble(
        n_neurons=n_neurons,
        dimensions=X_train.shape[1]
    )
    nengo.Connection(out_ensemble, recon_error, transform=-1)
    nengo.Connection(input, recon_error)
    nengo.Connection(
        recon_error, conn2.learning_rule
    )

    error_disp_node = nengo.Node(size_in=1)
    nengo.Connection(recon_error, error_disp_node, function=lambda x: np.sum(np.square(x)))

    # Input image display (for nengo_gui)
    image_shape = (1, 28, 28)
    display_func = image_display_function(image_shape, offset=1, scale=128)
    display_node = nengo.Node(display_func, size_in=input.size_out)
    nengo.Connection(input, display_node, synapse=None)

    output = nengo.Node(display_func, size_in=X_train.shape[1])
    nengo.Connection(out_ensemble, output, synapse=None)
    
    
