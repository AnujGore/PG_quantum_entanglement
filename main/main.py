from shadow import shadow_Schimdt
from Basis_measurement import basis_measurementList

snaps = 10
qubits = 2
size_of_set = 10

measurement_basis_list = basis_measurementList(snaps, qubits) #This is seeded 

dataset = [shadow_Schimdt(snaps, qubits, measurement_basis_list) for _ in range(size_of_set)]

