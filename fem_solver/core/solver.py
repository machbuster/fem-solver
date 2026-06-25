import numpy as np

def apply_boundary_conditions(K, f, fixed_dofs):
    """
    Dirichlet BC — penalty method.
    
    K          : (n,n) global stiffness matrix
    f          : (n,) force vector
    fixed_dofs : list of DOF indices to fix (displacement = 0)
    
    Returns K_mod, f_mod
    """
    K_mod = K.copy()
    f_mod = f.copy()
    penalty = 1e14 * np.max(np.diag(K))
    
    for dof in fixed_dofs:
        K_mod[dof, :] = 0
        K_mod[:, dof] = 0
        K_mod[dof, dof] = penalty
        f_mod[dof] = 0

    return K_mod, f_mod


def solve(K, f, fixed_dofs):
    """
    Linear static FEM solve.
    
    Returns u : (n,) displacement vector
    """
    K_mod, f_mod = apply_boundary_conditions(K, f, fixed_dofs)
    u = np.linalg.solve(K_mod, f_mod)
    return u