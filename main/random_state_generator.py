import numpy as np

class Basis_vectors:
    def __init__(self, list_of_basis_states, orthogonal = True) -> None:
        self.basis = list_of_basis_states
        self.__orthogonal_state = orthogonal
    
    def inner_product(self, basis1, basis2):
        if basis1 in self.basis:
            if basis2 in self.basis:
                if basis1 == basis2:
                    return 1
                elif self.__orthogonal_state:
                    return 0
                else: 
                    return np.NaN
                
class random_state_generator():

    def __init__(self, qubit) -> None:
        self.qubit = qubit
        state_whole = self.random_pure_state()
        self.basis = state_whole[1]
        self.state = state_whole[0]

    def random_pure_state(self):
        '''
        random_pure_state(n)

        - n is the dimensions of the Hilbert space

        returns: Single normalised (pure) state and the associated basis vectors (assumed to be orthogonal)
        '''
        n = 2*self.qubit

        mean = [0 for _ in range(n)]

        basis_vector = [i for i in range(n)]

        cov = []

        for one in range(n):
            cov_vec = [0 for _ in range(n)]
            cov_vec[one] = 1
            cov.append(cov_vec)

        return_state_unnormalized = np.random.multivariate_normal(mean, cov)
        normalisation_factor = np.linalg.norm(return_state_unnormalized)
        return_state_normalised = return_state_unnormalized/normalisation_factor

        return np.array(np.reshape(return_state_normalised, (int(len(return_state_normalised)/2), 2)), dtype = complex), Basis_vectors(basis_vector)
    
    def densityMat(self):
        '''
        Finds the density matrix of a state by doing the row and column wise multiplication of the state.

        Doing the column and row wise (conventional way) would return a single value.

        For all not possible values, a 0 value is returned.

        '''

        state = self.state.flatten()

        return_mat = np.zeros((len(state), len(state)), dtype= complex)

        basis = [i for i in range(len(state))]

        bra_state_vals = [np.conjugate(i) for i in state]
        bra = np.transpose(bra_state_vals)
        bra_matrix = np.zeros_like(return_mat)
        bra_matrix[:, -1] = bra

        ket_matrix = np.zeros_like(return_mat)
        ket_matrix[-1, :] = state

        self.return_mat = np.matmul(bra_matrix, ket_matrix, dtype=complex)

        return self.return_mat, Basis_vectors(basis)



def multiple_states(number_of_states, dimension):
    '''
    Based on random_pure_state(n)
    '''

    return_vector = []
    
    for new_state in range(number_of_states):
        return_vector.append(random_state_generator(dimension).random_pure_state()[0])

    basis_vector  = [i for i in range(dimension)]


    return return_vector, Basis_vectors(basis_vector)


density_mat = random_state_generator(2).densityMat()[0]

