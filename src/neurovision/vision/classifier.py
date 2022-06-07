"""
A Classifier for MNIST built with Nengo
"""

import nengo
from neurovision.data import MNIST


def build_classifier_net(n_neurons, data_path="mnist.npz"):
    """
    Builds a single layer classifier neural network for MNIST.

    :param n_neurons: Number of neurons.
    :param data_path: Path to MNIST labels

    :return model: Neural network
    """
    dataset = MNIST(data_path)
    n_dim = dataset.dim()

    model = nengo.Network(seed=3)
    solver = nengo.solvers.LstsqL2(reg=0.01)
    with model:
        img = nengo.Node(lambda t: dataset[int(t)][0], size_out=n_dim)
        a = nengo.Ensemble(
            n_neurons, n_dim,
            eval_points=dataset.get_eval_points(),
            neuron_type=nengo.LIF(),
            intercepts=nengo.dists.Choice([0.1]),
            max_rates=nengo.dists.Choice([100]),
        )

        v = nengo.Node(size_in=dataset.num_classes())

        nengo.Connection(img, a, synapse=None)
        nengo.Connection(
            a, v, 
            eval_points=dataset.get_eval_points(), 
            function=dataset.targets(), 
            solver=solver
        )

    return model
    