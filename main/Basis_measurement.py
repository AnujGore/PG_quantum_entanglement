import numpy as np

def basis_measurementList(shadow_snapshots, num_qubits):

    np.random.seed(666)

    basis_measurments = np.random.randint(0, 4, size = (int(1*shadow_snapshots), num_qubits))

    filter(lambda a: a!=[3 for _ in range(num_qubits)], basis_measurments)

    basis_measurments_as_string = np.array2string(basis_measurments)

    f = open("{}_basis_measurements.txt".format(shadow_snapshots), "w")
    f.write(basis_measurments_as_string)
    f.close()

    return basis_measurments