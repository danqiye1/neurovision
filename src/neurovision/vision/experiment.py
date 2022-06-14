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
from nengo_extras.vision import Gabor, Mask

model = nengo.Network()

rng = np.random.RandomState(9)

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(len(X_train), -1)
n_neurons=1000

with model:
    model.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
    model.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    model.config[nengo.Connection].synapse = None
    
    input = nengo.Node(lambda t: X_train[int(t), :])
    
    # Encoders are used as transforms for dimension reduction
    encoders = Mask((28,28)).populate(
        Gabor().generate(64, (11,11), rng=rng),
        rng=rng, 
        flatten=True
    )
    # Decoder transforms/weights for reconstruction
    decoders = Mask((28, 28)).populate(
        Gabor().generate(64, (11,11), rng=rng),
        rng=rng,
        flatten=True
    )
    
    in_ensemble = nengo.Ensemble(
        n_neurons=n_neurons,
        dimensions=X_train.shape[1]
    )

    hidden = nengo.Ensemble(
        n_neurons=n_neurons,
        dimensions=64
    )

    out_ensemble = nengo.Ensemble(
        n_neurons=n_neurons,
        dimensions=X_train.shape[1]
    )
    
    nengo.Connection(input, in_ensemble)
    conn1 = nengo.Connection(
        in_ensemble, hidden, 
        transform=encoders, 
        learning_rule_type=nengo.PES()
    )
    conn2 = nengo.Connection(
        hidden, out_ensemble, 
        transform=decoders.T,
        learning_rule_type=nengo.PES()
    )
    
    recon_error = nengo.Ensemble(
        n_neurons=n_neurons,
        dimensions=X_train.shape[1]
    )
    nengo.Connection(out_ensemble, recon_error)
    nengo.Connection(input, recon_error, transform=-1)
    nengo.Connection(
        recon_error, conn1.learning_rule, 
        function=lambda x: np.sum(np.square(x)) * np.ones(64)
    )
    nengo.Connection(
        recon_error, conn2.learning_rule,
        function=lambda x: np.sum(np.square(x)) * np.ones(X_train.shape[1])
    )
    
    
