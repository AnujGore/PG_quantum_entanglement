from Entanglement_shadow import classical_shadow, classical_shadow_manual
from Entanglement_classical import Entanglement_quantifier
from Basis_measurement import basis_measurementList

import numpy as np

import sys
sys.path.append("C:\\Users\\anuj_\\Documents\\UCL\\Individual Research Project\\PG_quantum_entanglement\\pennylane")

from pennylane import pennylane as qml

np.random.seed(666)

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

print(theta)

params = theta

# state = state_circuit(theta)

# basis_ids = basis_measurementList(num_snapshots, num_qubits)

# shadows_recipe = classical_shadow_manual(state, basis_ids, num_snapshots, num_qubits)

# from pennylane.measurements.expval import ExpectationMP

# @qml.qnode(dev)
# def shadow(recipe):
#     return_outcome = np.zeros_like(shadows_recipe, dtype = ExpectationMP)

#     for snap in range(len(recipe)):
#         for qubit in range(len(recipe[snap])):
#             return_outcome[snap][qubit] = qml.expval(shadows_recipe[snap][qubit]).eigvals

#     return return_outcome

# shadows = shadow(shadows_recipe)
# print(shadows)

shadows = classical_shadow(circuit, params, num_snapshots, num_qubits)
print(shadows)