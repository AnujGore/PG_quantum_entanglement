from Entanglement_shadow import classical_shadow
from Entanglement_classical import Entanglement_quantifier
from Basis_measurement import basis_measurementList

import numpy as np

import sys
sys.path.append("C:\\Users\\anuj_\\Documents\\UCL\\Individual Research Project\\PG_quantum_entanglement\\pennylane")

from pennylane import pennylane as qml

num_qubits = 2
wires = [i for i in range(num_qubits)]

dev = qml.device("default.qubit", wires=num_qubits)
@qml.qnode(dev)
def circuit(theta, **kwargs):
    observables = kwargs.pop("observables")
    qml.SpecialUnitary(theta, wires=wires)
    return [qml.expval(o) for o in observables]

theta = 2*np.pi*np.random.rand(4**num_qubits-1)

@qml.qnode(dev)
def state_circuit(theta):
    qml.SpecialUnitary(theta, wires= wires)
    return qml.state()

state = np.array(state_circuit(theta))
print(state)

eigen = Entanglement_quantifier.eigenvalues(state, (num_qubits, num_qubits))
print(Entanglement_quantifier.vonNeumann(eigen))

##Shadows below this in this bitch

snaps = 10
basis_measurements = basis_measurementList(snaps, num_qubits)

f = open("{}_basis_measurements.txt".format(snaps), "w")
f.write(np.array2string(basis_measurements))
f.close()

sdw = classical_shadow(circuit, theta, 10, num_qubits, basis_measurements)

print(sdw)
