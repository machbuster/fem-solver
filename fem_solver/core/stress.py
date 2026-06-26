import numpy as np
from fem_solver.elements.cst import CSTElement

def compute_stress(nodes, elements, u, E, nu, t=1.0):
    """
    Her eleman için stress ve Von Mises hesapla.
    
    nodes    : (n_nodes, 2) node koordinatları
    elements : (n_elems, 3) eleman node indeksleri
    u        : (2*n_nodes,) displacement vektörü
    
    Returns:
        stresses    : (n_elems, 3) — [sigma_x, sigma_y, tau_xy]
        von_mises   : (n_elems,)  — Von Mises stress
    """
    stresses = []
    von_mises = []

    for elem_nodes in elements:
        coords = nodes[elem_nodes]
        elem = CSTElement(coords, E, nu, t)
        
        B = elem.compute_B()
        D = elem.compute_D()

        # Eleman displacement vektörü
        dofs = []
        for i in elem_nodes:
            dofs += [2*i, 2*i+1]
        u_e = u[dofs]

        # Stress = D @ B @ u_e
        sigma = D @ B @ u_e
        stresses.append(sigma)

        # Von Mises stress
        sx, sy, txy = sigma
        vm = np.sqrt(np.maximum(sx**2 - sx*sy + sy**2 + 3*txy**2, 0))
        von_mises.append(vm)

    return np.array(stresses), np.array(von_mises)