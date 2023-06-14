# import sys
# sys.path.append("C:\\Users\\anuj_\\Documents\\UCL\\Individual Research Project\\PG_quantum_entanglement\\pennylane")

# from pennylane import pennylane as qml

# def classical_shadow(activated_qubit, params, shadow_size, num_qubits, basis_measurements):
#     """
#     Given how a state is prepared via the activated qubit function,
#     this method finds the Pauli X, Y and Z values of the activated qubit
#     repeatedly and returns the results which are the shadows

#     Args:
#         activated_qubit (function): A Pennylane QNode
#         params (array): The circuit parameter
#         shadow_size (int): Number of times the Pauli X Y and Z values are randomly selected to calculate
#         num_qubits (int): Number of qubits in the circuit

#     Returns:
#         Shadow array: The values of the Pauli X Y and Z
#     """

#     unitary_ensenmble = [qml.PauliX, qml.PauliY, qml.PauliZ, qml.Identity]

#     outcomes = np.zeros_like(basis_measurements)

#     for ns in range(shadow_size):
#         obs = [unitary_ensenmble[int(basis_measurements[ns, i])](i) for i in range(num_qubits)]
#         outcomes[ns, :] = activated_qubit(params, observables = obs)

#     return outcomes

import numpy as np

def to_euler(number):
    alpha = np.real(number)
    beta = np.imag(number)

    hypt = np.sqrt(alpha**2 + beta**2)

    if alpha == 0:
        return np.arccos(beta/hypt)/(np.pi/2)
    else:
        return np.arcsin(alpha/hypt)/(np.pi/2)


def expval(basis, state):
    state_conj = np.conjugate(state)
    bra_matrix_prod = state_conj @ basis
    return to_euler(bra_matrix_prod @ state)

def classical_shadow_manual(state, basis_measurements, snapshots, num_qubits):

    pauli_x = np.array([[0.+0.j,1.+0.j],[1.+0.j, 0.+0.j]], dtype=complex)
    pauli_y = np.array([[0.+0.j,0.+1.j],[0.+1.j, 0.+0.j]], dtype=complex)
    pauli_z = np.array([[1.+0.j,0.+0.j],[0.+0.j, -1.+0.j]], dtype=complex)
    identity = np.array([[1.+0.j,0.+0.j],[0.+0.j, 1.+0.j]], dtype=complex)

    unitary_ensemble = [pauli_x, pauli_y, pauli_z, identity]

    outcomes = np.zeros_like(basis_measurements)
    
    for i in range(snapshots):
        basis = basis_measurements[i]
        for qubit_no in range(num_qubits):
            basis_qubit = basis[qubit_no]
            pauli_basis = unitary_ensemble[basis_qubit]
            qubit = state[2*qubit_no:2*qubit_no+2]
            outcomes[i][qubit_no] = expval(pauli_basis, qubit)

            
    return outcomes
