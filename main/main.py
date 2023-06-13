from random_state_generator import random_state_generator
from Entanglement_shadow import classical_shadow
from Entanglement_classical import Entanglement_quantifier

import numpy as np

import sys
sys.path.append("C:\\Users\\anuj_\\Documents\\UCL\\Individual Research Project\\PG_quantum_entanglement\\pennylane")

from pennylane import pennylane as qml

qubits = 2

dev = qml.device("default.mixed", wires=int(np.log2(qubits) + 1))
@qml.qnode(dev)
def circuit(rho, **kwargs):
    observables = kwargs.pop("observable")
    qml.QubitDensityMatrix(rho, wires=[i for i in range(int(np.log2(qubits))+1)])
    return [qml.expval(o) for o in observables]

new_state = random_state_generator(qubits)

rho = new_state.densityMat()[0]

print(rho)
