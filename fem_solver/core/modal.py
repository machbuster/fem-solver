import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from fem_solver.elements.cst import CSTElement

def compute_mass_matrix(nodes, elements, rho, t=1.0):
    """
    Consistent mass matrix assembly.
    
    nodes    : (n_nodes, 2)
    elements : (n_elems, 3)
    rho      : malzeme yoğunluğu (kg/m³)
    t        : kalınlık (m)
    
    Returns M : (2*n_nodes, 2*n_nodes)
    """
    n_nodes = len(nodes)
    M = np.zeros((2*n_nodes, 2*n_nodes), dtype=np.float64)

    for elem_nodes in elements:
        coords = nodes[elem_nodes]
        x1,y1 = coords[0]
        x2,y2 = coords[1]
        x3,y3 = coords[2]
        A = 0.5 * abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))

        # Consistent mass matrix for CST (12x12 before reduction)
        # Me = rho * t * A / 12 * [2,0,1,0,1,0; ...]
        c = rho * t * A / 12.0
        Me = c * np.array([
            [2,0,1,0,1,0],
            [0,2,0,1,0,1],
            [1,0,2,0,1,0],
            [0,1,0,2,0,1],
            [1,0,1,0,2,0],
            [0,1,0,1,0,2],
        ])

        dofs = []
        for i in elem_nodes:
            dofs += [2*i, 2*i+1]

        for i, gi in enumerate(dofs):
            for j, gj in enumerate(dofs):
                M[gi, gj] += Me[i, j]

    return M


def modal_analysis(K, M, fixed_dofs, n_modes=6):
    """
    Modal analiz — doğal frekanslar ve mod şekilleri.
    
    K         : global stiffness matrix
    M         : global mass matrix
    fixed_dofs: sabit DOF indeksleri
    n_modes   : hesaplanacak mod sayısı
    
    Returns:
        freqs     : (n_modes,) doğal frekanslar (Hz)
        mode_shapes: (n_dofs, n_modes) mod şekilleri
    """
    n = K.shape[0]
    free_dofs = [i for i in range(n) if i not in set(fixed_dofs)]

    K_free = csr_matrix(K[np.ix_(free_dofs, free_dofs)])
    M_free = csr_matrix(M[np.ix_(free_dofs, free_dofs)])

    n_modes = min(n_modes, len(free_dofs) - 2)

    eigenvalues, eigenvectors = eigsh(K_free, k=n_modes, M=M_free, sigma=0, which='LM')

    # Doğal frekanslar (Hz)
    freqs = np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)
    freqs = np.sort(freqs)

    # Tam mod şekilleri (sabit DOF'lar sıfır)
    mode_shapes = np.zeros((n, n_modes))
    for i, dof in enumerate(free_dofs):
        mode_shapes[dof, :] = eigenvectors[i, :]

    return freqs, mode_shapes