from Entanglement_shadow import classical_shadow
from Entanglement_classical import Entanglement_quantifier
from scipy import sparse

import numpy as np

import sys
sys.path.append("C:\\Users\\anuj_\\Documents\\UCL\\Individual Research Project\\PG_quantum_entanglement\\pennylane")

from pennylane import pennylane as qml

def shadow_Schimdt(num_snapshots, num_qubits, measurement_basis_list):
    """
    This method first creates a random state via Special Unitary transformations. This states' Schmidt Gap is then
    calculated via our Entanglement_classical class. The shadows are then found and stored in a 4 x num_snapshots array.
    For the measurement basis ID taken, this array stores the shadows value (+/- 1).

    Args:
        - num_snapshots (int): Number of snapshots for the shadow
        - num_qubits (int): Number of qubits

    Returns:
        - 2d array of shadows, Schmidt Gap } as one variable
    """

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

    params = theta

    state_coeff = state_circuit(theta)

    eiges = Entanglement_quantifier.eigenvalues(state_coeff, (num_qubits, num_qubits))
    Schmidt_gap = Entanglement_quantifier.schmidtGap(eiges)

    shadows = classical_shadow(circuit, params, measurement_basis_list, num_snapshots, num_qubits)

    #To make a machine readable training set, we will create a 4xnum_snapshots array to store the shadows.
    #Unselected measurement basis ids will have 0 or NaN values (need to see if networks can operate with NaN values)

    shadow_states = np.zeros((num_snapshots, 4))
    for j, row in enumerate(measurement_basis_list):
        for i, basis in enumerate(row):
            shadow_states[j][basis] = shadows[j][i]

    shadow_states = sparse.csr_matrix(shadow_states) #Compressing the matrix

    return (shadow_states, Schmidt_gap)





