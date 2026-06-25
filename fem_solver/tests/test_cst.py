import numpy as np
import sys
sys.path.insert(0, '/workspaces/fem-solver')

from fem_solver.elements.cst import CSTElement

def test_area():
    nodes = np.array([[0,0],[1,0],[0,1]], dtype=float)
    elem = CSTElement(nodes, E=210e9, nu=0.3)
    assert abs(elem.area() - 0.5) < 1e-10

def test_Ke_shape():
    nodes = np.array([[0,0],[1,0],[0,1]], dtype=float)
    elem = CSTElement(nodes, E=210e9, nu=0.3)
    Ke = elem.compute_Ke()
    assert Ke.shape == (6,6)

def test_Ke_symmetric():
    nodes = np.array([[0,0],[1,0],[0,1]], dtype=float)
    elem = CSTElement(nodes, E=210e9, nu=0.3)
    Ke = elem.compute_Ke()
    assert np.allclose(Ke, Ke.T)

if __name__ == "__main__":
    test_area()
    test_Ke_shape()
    test_Ke_symmetric()
    print("Tüm testler geçti!")