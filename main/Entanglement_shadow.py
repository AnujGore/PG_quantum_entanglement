import sys
sys.path.append("C:\\Users\\anuj_\\Documents\\UCL\\Individual Research Project\\PG_quantum_entanglement\\pennylane")

from pennylane import pennylane as qml
import numpy as np

from numba import prange

np.random.seed(666)

def letter_val (num_list):
    return_list = []
    for i in prange(len(num_list)):
        if num_list[i] == 0: return_list.append("X")
        elif num_list[i] == 1: return_list.append("Y")
        elif num_list[i] == 2: return_list.append("Z")

    return return_list

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

    outcomes = np.zeros_like(np.column_stack((unitary_picks, unitary_picks)))

    for ns in prange(shadow_size):
        obs = [unitary_ensenmble[int(unitary_picks[ns, i])](i) for i in range(num_qubits)]
        outcomes[ns, :num_qubits] = activated_qubit(params, observable = obs)
        outcomes[ns, num_qubits:] = unitary_picks[ns]

    return(outcomes)

num_qubits = 1

dev = qml.device("default.qubit", wires = num_qubits, shots = 1)

@qml.qnode(dev)
def local_rotation(params, **kwargs):
    observables = kwargs.pop("observable")
    for w in dev.wires:
        qml.RX(params[w][0], wires = w)
        qml.RY(params[w][1], wires = w)
        qml.RZ(params[w][2], wires = w)
    
    return [qml.expval(o) for o in observables]

params = np.random.rand(num_qubits,3)
print(params)
num_snaps = 10000
shadow = classical_shadow(local_rotation, params, num_snaps, num_qubits)
shadow_shadow = shadow[:, :num_qubits]
shadow_measure = shadow[:, num_qubits:]

def Bloch_estimate(shadows, measure):
    """
    Find the projection of each measurement onto the Bloch Sphere. 
    Theoretically, we should be able to recreate the qubit from this sphere

    Args:
        shadows (array): Array of shadows seen from the shadow calculation
        measure (array): Measurement axis (Pauli measurement)
    
    Returns:
        Rotatory List
    """

    num_qubits = len(shadows[0])
    if TypeError: num_qubits = 1
    num_iters = len(shadows)

    qubit_vals = []
    
    for qubit in range(num_qubits):
        Pauli_X = 0; Pauli_Y = 0; Pauli_Z = 0
        for outcome in prange(num_iters):
            measurement = measure[outcome]
            if measurement == 0: Pauli_X += shadows[outcome]
            elif measurement == 1: Pauli_Y += shadows[outcome]
            elif measurement == 2: Pauli_Z += shadows[outcome]

        qubit_vals.append([Pauli_X/num_iters, Pauli_Y/num_iters, Pauli_Z/num_iters])

    return qubit_vals

print(Bloch_estimate(shadow_shadow, shadow_measure))