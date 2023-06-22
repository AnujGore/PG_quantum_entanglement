from Entanglement_shadow import classical_shadow, classical_shadow_manual
from Entanglement_classical import Entanglement_quantifier
from Basis_measurement import basis_measurementList

import numpy as np

import sys
sys.path.append("C:\\Users\\anuj_\\Documents\\UCL\\Individual Research Project\\PG_quantum_entanglement\\pennylane")

from pennylane import pennylane as qml

num_qubits = 2

dev = qml.device("default.qubit", wires = num_qubits)
wires = [i for i in range(num_qubits)]

theta = 2*np.pi*np.random.rand(4**num_qubits-1)

@qml.qnode(dev)
def state_circuit(theta):
    qml.SpecialUnitary(theta, wires= wires)
    return qml.state()

@qml.qnode(dev)
def circuit(theta, **kwargs):
    observables = kwargs.pop("observable")
    qml.SpecialUnitary(theta, wires= wires)
    return [qml.expval(o) for o in observables]

num_snapshots = 10

params = theta

state = state_circuit(theta)

basis_ids = basis_measurementList(num_snapshots, num_qubits)

eigens = Entanglement_quantifier.eigenvalues(state, (num_qubits, num_qubits))
print(Entanglement_quantifier.vonNeumann(eigens))

shadows_manual = classical_shadow_manual(state, basis_ids, num_snapshots, num_qubits)
print(shadows_manual)

shadows = classical_shadow(circuit, params, num_snapshots, num_qubits)
# print(shadows)