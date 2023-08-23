import numpy as np
import os.path

def basis_measurementList(shadow_snapshots, num_qubits):
    fname = "{}_basis_measurements_{}_qubits.txt".format(shadow_snapshots, num_qubits)
    # if os.path.isfile(fname):
    #     basis_measurments = [[int(float(val)) for val in line.strip('\n').replace('[', '').replace('.','').replace(']', '').replace(' ', '')] for line in open(fname)]
    #     basis_measurments = np.reshape(basis_measurments, (int(shadow_snapshots), num_qubits))
    #     return basis_measurments
    # else:
    #     pass

    basis_measurments = np.random.randint(0, 4, size = (int(1*shadow_snapshots), num_qubits))

    filter(lambda a: a!=[3 for _ in range(num_qubits)], basis_measurments)

    basis_measurments_as_string = np.array2string(basis_measurments)

    f = open(fname, "w")
    f.write(basis_measurments_as_string)
    f.close()

    return basis_measurments