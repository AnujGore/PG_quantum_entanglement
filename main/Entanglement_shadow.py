import sys
sys.path.append("C:\\Users\\anuj_\\Documents\\UCL\\Individual Research Project\\PG_quantum_entanglement\\pennylane")

from pennylane import pennylane as qml

import numpy as np

def classical_shadow(activated_qubit, params, shadow_size, num_qubits):
    """
    Given how a state is prepared via the activated qubit function,
    this method finds the Pauli X, Y and Z values of the activated qubit
    repeatedly and returns the results which are the shadows

    Args:
        activated_qubit (function): A Pennylane QNode
        params (array): The circuit parameter
        shadow_size (int): Number of times the Pauli X Y and Z values are randomly selected to calculate
        num_qubits (int): Number of qubits in the circuit

    Returns:
        Shadow array: The values of the Pauli X Y and Z
    """

    unitary_ensenmble = [qml.PauliX, qml.PauliY, qml.PauliZ]

    unitary_ids = np.random.randint(0, 3, size = (shadow_size, num_qubits))
    outcomes = np.zeros_like(unitary_ids)

    for ns in range(shadow_size):
        obs = [unitary_ensenmble[int(unitary_ids[ns, i])](i) for i in range(num_qubits)]
        outcomes[ns, :] = activated_qubit(params, observable = obs)

    return outcomes

# def to_euler(number):
#     alpha = np.real(number)
#     beta = np.imag(number)

#     hypt = np.sqrt(alpha**2 + beta**2)

#     if alpha == 0:
#         return np.arccos(beta/hypt)/(np.pi/2)
#     else:
#         return np.arcsin(alpha/hypt)/(np.pi/2)


# def expval(basis, state):
#     state_conj = np.conjugate(state)
#     bra_matrix_prod = state_conj @ basis
#     return np.real(bra_matrix_prod @ state)

def classical_shadow_manual(state, basis_measurements, snapshots, num_qubits):

    # pauli_x = np.array([[0.+0.j,1.+0.j],[1.+0.j, 0.+0.j]], dtype=complex)
    # pauli_y = np.array([[0.+0.j,0.-1.j],[0.+1.j, 0.+0.j]], dtype=complex)
    # pauli_z = np.array([[1.+0.j,0.+0.j],[0.+0.j, -1.+0.j]], dtype=complex)
    # identity = np.array([[1.+0.j,0.+0.j],[0.+0.j, 1.+0.j]], dtype=complex)

    pauli_x = qml.PauliX
    pauli_y = qml.PauliY
    pauli_z = qml.PauliZ
    identity = qml.Identity

    unitary_ensemble = [pauli_x, pauli_y, pauli_z, identity]

    outcomes = np.zeros_like(basis_measurements, dtype=qml.operation.Observable)
    
    for i in range(snapshots):
        basis = basis_measurements[i]
        for qubit_no in range(num_qubits):
            basis_qubit = basis[qubit_no]
            pauli_basis = unitary_ensemble[basis_qubit]
            # qubit = state[2*qubit_no:2*qubit_no+2]
            outcomes[i][qubit_no] = pauli_basis(qubit_no)

            
    return outcomes
