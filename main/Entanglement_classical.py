import numpy as np
import pytest

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

def test_bell_state():
    eigen_bell = Entanglement_quantifier.eigenvalues(bell_state, bell_size)
    assert Entanglement_quantifier.schmidtGap(eigen_bell) == 0.0
    assert Entanglement_quantifier.vonNeumann(eigen_bell) == 1.0

def test_pure_0_state():
    eigen_pure = Entanglement_quantifier.eigenvalues(pure_state, pure_size)
    assert Entanglement_quantifier.schmidtGap(eigen_pure) == 1.0
    assert Entanglement_quantifier.vonNeumann(eigen_pure) == 0.0




