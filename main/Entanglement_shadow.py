import sys
sys.path.append("C:\\Users\\anuj_\\Documents\\UCL\\Individual Research Project\\PG_quantum_entanglement\\pennylane")

from pennylane import pennylane as qml
import pennylane.numpy as np

np.random.seed(666)

num_qubits = 2

dev = qml.device("default.qubit", wires=1, shots=1)

params = np.random.randn(num_qubits)
params = [0, 0]
print(params)

@qml.qnode(dev, interface = "autograd")
def q_circuit(params):
    qml.RX(params[0], wires = 0)
    qml.RY(params[1], wires = 0)
    return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]

unitary_ensenmble = [4, 6, 9]
unitary_ids = np.random.randint(0, 3, size =(10, 3))
outcomes = np.zeros((10, 3))

for ns in range(10):
    obs = [unitary_ensenmble[int(unitary_ids[ns, i])] for i in range(3)]
    outcomes[ns, :] = obs

print(outcomes)



