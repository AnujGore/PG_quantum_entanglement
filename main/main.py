from random_state_generator import random_state_generator
from Entanglement_shadow import classical_shadow
from Entanglement_classical import Entanglement_quantifier

import numpy as np

import sys
sys.path.append("C:\\Users\\anuj_\\Documents\\UCL\\Individual Research Project\\PG_quantum_entanglement\\pennylane")

from pennylane import pennylane as qml

qubits = 2
size =(2, 2)

dev = qml.device("default.mixed", wires=int(np.log2(qubits)))
@qml.qnode(dev)
def circuit(rho):
    qml.QubitDensityMatrix(rho, wires=[i for i in range(int(np.log2(qubits)))])
    return qml.state()

new_state = random_state_generator(qubits)

state = new_state.state

print("State is: {}".format(state))

eigenVals = Entanglement_quantifier.eigenvalues(state)
print("Eigenvalues after SVD: {}".format(eigenVals))
print("Schmidt Gap: {}".format(Entanglement_quantifier.schmidtGap(eigenVals)))
print("Von Neumann: {}".format(Entanglement_quantifier.vonNeumann(eigenVals)))
