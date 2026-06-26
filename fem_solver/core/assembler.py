import numpy as np
from fem_solver.elements.cst import CSTElement

def assemble(nodes, elements, E, nu, t=1.0):
    """
    Global stiffness matrix assembly.
    
    nodes    : (n_nodes, 2) array — tüm node koordinatları
    elements : (n_elems, 3) array — her satır 3 node indeksi
    E        : Young's modulus
    nu       : Poisson's ratio
    t        : thickness
    
    Returns K : (2*n_nodes, 2*n_nodes) global stiffness matrix
    """
    n_nodes = len(nodes)
    K = np.zeros((2*n_nodes, 2*n_nodes), dtype=np.float64)
    for elem_nodes in elements:
        coords = nodes[elem_nodes]
        elem = CSTElement(coords, E, nu, t)
        Ke = elem.compute_Ke()

        # DOF mapping: node i → [2i, 2i+1]
        dofs = []
        for i in elem_nodes:
            dofs += [2*i, 2*i+1]

        for i, gi in enumerate(dofs):
            for j, gj in enumerate(dofs):
                K[gi, gj] += Ke[i, j]

    # Isolated node'ları sabitle
    for i in range(2*n_nodes):
        if K[i, i] == 0.0:
            K[i, i] = 1.0

    return K