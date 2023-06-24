from Entanglement_shadow import classical_shadow, classical_shadow_manual
from Entanglement_classical import Entanglement_quantifier
from Basis_measurement import basis_measurementList

import numpy as np

import sys
sys.path.append("C:\\Users\\anuj_\\Documents\\UCL\\Individual Research Project\\PG_quantum_entanglement\\pennylane")

from pennylane import pennylane as qml

num_qubits = 2

dev = qml.device("default.qubit", wires = num_qubits, shots = 1)
wires = [i for i in range(num_qubits)]

theta = 2*np.pi*np.random.rand(4**num_qubits-1)

# theta = np.random.randn(num_qubits)

@qml.qnode(dev)
def state_circuit(theta):
    qml.SpecialUnitary(theta, wires= wires)
    return qml.state()

@qml.qnode(dev)
def circuit(theta, **kwargs):
    observables = kwargs.pop("observable")
    qml.SpecialUnitary(theta, wires= wires)
    return [qml.expval(o) for o in observables]

num_snapshots = 1000

params = theta

state = state_circuit(theta)
print(state)

shadows = classical_shadow(circuit, params, num_snapshots, num_qubits)
shadows_sum = [[i for i in shadows[0][:, 0]],[i for i in shadows[0][:, 1]]]
shadows_sum = [sum(shadows_sum[0]), sum(shadows_sum[1])]

