import numpy as np

def basis_measurementList(shadow_snapshots, num_qubits):

    np.random.seed(666)

    basis_measurments = np.random.randint(0, 4, size = (int(1.2*shadow_snapshots), num_qubits))

    filter(lambda a: a!=[3 for _ in range(num_qubits)], basis_measurments)

    return basis_measurments