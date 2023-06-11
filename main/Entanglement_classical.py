#First we need to create a set of random pure states. This can be done by creating a set of vectors with n dimensions
#Which is normalized to 1.

import numpy as np

def random_pure_state(n):
    '''
    random_pure_state(n)

     - n is the dimensions of the Hilbert space

    returns: Single normalised (pure) state.
    '''

    mean = [0 for _ in range(n)]

    cov = []

    for one in range(n):
        cov_vec = [0 for _ in range(n)]
        cov_vec[one] = 1
        cov.append(cov_vec)

    return_state_unnormalized = np.random.multivariate_normal(mean, cov)
    normalisation_factor = np.linalg.norm(return_state_unnormalized)
    return_state_normalised = return_state_unnormalized/normalisation_factor

    return return_state_normalised

#Now we need to generate m number of random states to evaluate their Schmidt Gap

def random_states_generator(number_of_states, dimension):
    '''
    Based on random_pure_state(n)
    '''

    return_vector = []

    for new_state in range(number_of_states):
        return_vector.append(random_pure_state(dimension))


    return return_vector

#Now we need to calculate the Schmidt Gap between each state

def densityMat(state):
    '''
    Finds the density matrix of a state by doing the row and column wise multiplication of the state.

    Doing the column and row wise (conventional way) would return a single value.

    For all not possible values, a 0 value is returned.

    '''
    return_mat = np.zeros((len(state), len(state)))
    try:
        for row in range(len(state)):
            for col in range(len(state)):
                return_mat[row][col] = state[row]*state[col]
    except:
        pass

    return return_mat

#Now we need to find the Schmidt gap classically
#For that first, we find the SVD of the density matrix

class Entanglement_quantifier:
    def __init__(self) -> None:
        pass


    def eigenvalues(states, size):
        '''
        SVD_quant quantifies the eigenvalue matrix of the SVD of a state

        The input must be greater than 3 and not a prime number as the product of the number of rows and columns 
        is the length of the states.
        '''

        rows = size[0]
        cols = size[1]
        
        reshape_array = (rows, cols)
        reshape_state= np.reshape(states, reshape_array)
        a, b, c = np.linalg.svd(reshape_state)
        return b

    def vonNeumann(b):
        ent = 0
        
        for eigen in b:
            if eigen == 0:
                ent -= 0
            else:
                ent -= eigen**2*np.log2(eigen**2)
        return ent
        
    def schmidtGap(b):
        return (b[0]- b[1])
        

bell_state = [1/2, 0, 0, 1/2] #circle with line thru it with a positive sign
bell_size = (2, 2)

pure_state = [1, 0, 0, 0]
pure_size = (2, 2)

measurement_size = 16
Test_states = random_pure_state(measurement_size)
test_size = (2, 8)

eigen = Entanglement_quantifier.eigenvalues(Test_states, test_size)
print("Measurement size: {}; qudit: {}; number of qudits: {}".format(measurement_size, test_size[0], test_size[1]))
print("Schmidt Gap: ", Entanglement_quantifier.schmidtGap(eigen))
print("Von Neumann: ", Entanglement_quantifier.vonNeumann(eigen))



