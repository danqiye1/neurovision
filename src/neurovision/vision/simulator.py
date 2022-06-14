"""
Boilerplate for putting network on NengoGUI for visualisation and simulation.
"""

import nengo
import numpy as np
import tensorflow as tf
from nengo_extras.data import one_hot_from_labels
from nengo_extras.vision import Gabor, Mask

model = nengo.Network()

rng = np.random.RandomState(9)

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(len(X_train), -1)
T_train = one_hot_from_labels(y_train)
n_neurons=1000

with model:
    model.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
    model.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    model.config[nengo.Connection].synapse = None
    
    input = nengo.Node(lambda t: X_train[int(t), :])
    gt = nengo.Node(lambda t: y_train[int(t)])
    
    pre = nengo.Ensemble(
        n_neurons=n_neurons,
        dimensions=X_train.shape[1],
        eval_points=X_train
    )
    
    encoders = Gabor().generate(n_neurons, (11,11), rng=rng)
    encoders = Mask((28,28)).populate(encoders, rng=rng, flatten=True)
    
    pre.encoders=encoders
    
    v = nengo.Node(size_in=T_train.shape[1])
    
    
    nengo.Connection(input, pre)
    conn = nengo.Connection(pre, v, eval_points=X_train, function=T_train)
    
    
    
    
