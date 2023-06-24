from Entanglement_shadow import classical_shadow
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
print(state)

eiges = Entanglement_quantifier.eigenvalues(state, (num_qubits, num_qubits))
print(Entanglement_quantifier.schmidtGap(eiges))
print(Entanglement_quantifier.vonNeumann(eiges))


measurement_basis_list = basis_measurementList(num_snapshots, num_qubits) #This is seeded 

shadows = classical_shadow(circuit, params, measurement_basis_list, num_snapshots, num_qubits)
print(shadows)

