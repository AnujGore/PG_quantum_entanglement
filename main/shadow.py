from Entanglement_shadow import classical_shadow
from Entanglement_classical import Entanglement_quantifier
from scipy import sparse

from Basis_measurement import basis_measurementList

import numpy as np
import matplotlib.pyplot as plt

import pennylane as qml

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

    #4 different arrays for each observable with the length being number of snaps and width being number of qubits

    sigma_x = np.zeros((num_snapshots, num_qubits))
    sigma_y = np.zeros_like(sigma_x)
    sigma_z = np.zeros_like(sigma_y)
    identity = np.zeros_like(sigma_z)

    for i, row in enumerate(measurement_basis_list):
        for j, basis in enumerate(row):
            if basis == 0: sigma_x[i][j] = shadows[i][j]
            elif basis == 1: sigma_y[i][j] = shadows[i][j]
            elif basis == 2: sigma_z[i][j] = shadows[i][j]
            elif basis == 3: identity[i][j] = shadows[i][j]

    #shadow_states = [sparse.csr_matrix(sigma_x), sparse.csr_matrix(sigma_y), sparse.csr_matrix(sigma_z), sparse.csr_matrix(identity)]

    shadow_states = [sigma_x, sigma_y, sigma_z, identity]

    return (shadow_states, Schmidt_gap)

def create_dataset(snaps, qubits, length):
    """
    Creates the dataset based on the shadow_Schmidt function

    Args:
        - snaps (int): Number of snapshots
        - qubits (int): Number of qubits
        - length (int): Size of dataset (number of times the simulation runs and creates different entanglements)
    """
    measurement_basis_list = basis_measurementList(snaps, qubits)

    dataset = [shadow_Schimdt(snaps, qubits, measurement_basis_list) for _ in range(length)]

    shadows = [np.array(dataset[val][0]) for val in range(len(dataset))]
    entropy = [dataset[val][1] for val in range(len(dataset))]

    return shadows, entropy


def three_D_model(shadows, entropy):
    """
    Uses matplotlib to visualize the 3D transformation of the data to preserve basis context

    Args:
        - shadows (array): Shadows of quantum system
        - entropy (float): Entanglement entropy of system (could be schmidt gap or anything used to quantify entanglement)

    Returns:
        matplotlib.pyplot.plt
        (Need to explicize plt.show() to visualize)
    """
    ones = []
    neg_ones = []

    for i in range(len(shadows)):
        plane = shadows[i]
        for j in range(len(plane)):
            iter = plane[j]
            for k in range(len(iter)):
                data = iter[k]
                if data == 1.0 : ones.append([i, j, k])
                if data == -1.0: neg_ones.append([i, j, k])


    ax = plt.figure().add_subplot(projection='3d')
    for points in ones:
        ax.scatter(points[0], points[1], points[2], marker = 'o', color = 'red')

    for points in neg_ones:
        ax.scatter(points[0], points[1], points[2], marker = '^', color = 'green')

    ax.set_xlabel("Observable plane")
    ax.set_ylabel("Shadow")
    ax.set_zlabel("Qubit")
    plt.title(entropy)
    
    return plt



