"""
Boilerplate for putting network on NengoGUI for visualisation and simulation.
"""

import nengo
from .classifier import build_classifier_net

model = nengo.Network()

with model:
    net = build_classifier_net(1000)
