import sys
sys.path.append("C:\\Users\\anuj_\\Documents\\UCL\\Individual Research Project\\PG_quantum_entanglement\\pennylane")

from pennylane import pennylane as qml
import numpy as np

from numba import prange

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

    unitary_picks = np.random.randint(0, 3, size = (shadow_size, num_qubits)) #The picks must be for the number of
    #shadows and the number of qubits

    outcomes = np.zeros_like(unitary_picks)

    for ns in prange(shadow_size):
        obs = [unitary_ensenmble[int(unitary_picks[ns, i])](i) for i in range(num_qubits)]
        outcomes[ns, :] = activated_qubit(params, observable = obs)

    return(outcomes, unitary_picks)