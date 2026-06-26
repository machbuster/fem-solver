import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def apply_boundary_conditions(K, f, fixed_dofs):
    K_mod = K.copy()
    f_mod = f.copy()
    penalty = 1e14 * K.diagonal().max()
    for dof in fixed_dofs:
        K_mod[dof, :] = 0
        K_mod[:, dof] = 0
        K_mod[dof, dof] = penalty
        f_mod[dof] = 0
    return K_mod, f_mod

def solve(K, f, fixed_dofs):
    n = K.shape[0]
    # Elimination method — daha kararlı
    free_dofs = [i for i in range(n) if i not in set(fixed_dofs)]
    
    K_free = K[np.ix_(free_dofs, free_dofs)]
    f_free = f[free_dofs]
    
    from scipy.sparse import csr_matrix
    K_sparse = csr_matrix(K_free)
    u_free = spsolve(K_sparse, f_free)
    
    u = np.zeros(n)
    for i, dof in enumerate(free_dofs):
        u[dof] = u_free[i]
    
    return u