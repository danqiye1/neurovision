"""
A Nengo Vision Model for MNIST
"""

import nengo
from mnist import MNIST
from matplotlib import pyplot as plt

from pdb import set_trace as bp

dataset = MNIST()
n_neurons = 1000
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
    conn = nengo.Connection(
        a, v, 
        eval_points=dataset.get_eval_points(), 
        function=dataset.targets(), 
        solver=solver
    )

    probe_out = nengo.Probe(v, synapse=0.01)

with nengo.Simulator(model) as sim:
    sim.run(10)

plt.plot(sim.trange(), sim.data[probe_out])
plt.savefig("mnist.png")