import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

def compute_thermal_stiffness(nodes, elements, k, t=1.0):
    """
    Thermal stiffness matrix (conductance matrix) assembly.
    
    nodes    : (n_nodes, 2)
    elements : (n_elems, 3)
    k        : termal iletkenlik (W/m·K)
    t        : kalınlık (m)
    
    Returns K_th : (n_nodes, n_nodes)
    """
    n_nodes = len(nodes)
    K_th = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    for elem_nodes in elements:
        coords = nodes[elem_nodes]
        x1,y1 = coords[0]
        x2,y2 = coords[1]
        x3,y3 = coords[2]

        A = 0.5 * abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))

        # Shape function gradients
        b = np.array([y2-y3, y3-y1, y1-y2])
        c = np.array([x3-x2, x1-x3, x2-x1])

        # Element conductance matrix
        Ke = k * t * A / (4 * A**2) * (np.outer(b, b) + np.outer(c, c))

        for i, gi in enumerate(elem_nodes):
            for j, gj in enumerate(elem_nodes):
                K_th[gi, gj] += Ke[i, j]

    return K_th


def solve_thermal(K_th, nodes, hot_edge, cold_edge, T_hot, T_cold):
    """
    Steady-state thermal analiz.
    
    K_th     : thermal stiffness matrix
    nodes    : (n_nodes, 2)
    hot_edge : 'left','right','top','bottom' — sıcak kenar
    cold_edge: 'left','right','top','bottom' — soğuk kenar
    T_hot    : sıcak kenar sıcaklığı (°C)
    T_cold   : soğuk kenar sıcaklığı (°C)
    
    Returns T : (n_nodes,) sıcaklık dağılımı
    """
    n = len(nodes)
    f = np.zeros(n)

    x_max = nodes[:,0].max(); x_min = nodes[:,0].min()
    y_max = nodes[:,1].max(); y_min = nodes[:,1].min()
    tol_x = (x_max - x_min) * 0.01
    tol_y = (y_max - y_min) * 0.01

    def get_edge(edge):
        if edge == "left":   return [i for i,n in enumerate(nodes) if n[0] < x_min + tol_x]
        elif edge == "right": return [i for i,n in enumerate(nodes) if n[0] > x_max - tol_x]
        elif edge == "bottom":return [i for i,n in enumerate(nodes) if n[1] < y_min + tol_y]
        elif edge == "top":   return [i for i,n in enumerate(nodes) if n[1] > y_max - tol_y]
        return []

    K_mod = K_th.copy()
    penalty = 1e14 * np.max(np.abs(np.diag(K_th)))

    for n_idx in get_edge(hot_edge):
        K_mod[n_idx, :] = 0
        K_mod[:, n_idx] = 0
        K_mod[n_idx, n_idx] = penalty
        f[n_idx] = penalty * T_hot

    for n_idx in get_edge(cold_edge):
        K_mod[n_idx, :] = 0
        K_mod[:, n_idx] = 0
        K_mod[n_idx, n_idx] = penalty
        f[n_idx] = penalty * T_cold

    # Isolated node kontrolü
    for i in range(n):
        if K_mod[i, i] == 0:
            K_mod[i, i] = 1.0

    T = spsolve(csr_matrix(K_mod), f)
    return T


def compute_heat_flux(nodes, elements, T, k):
    """
    Her eleman için ısı akısı hesapla.
    
    Returns:
        flux_x : (n_elems,) x yönü ısı akısı
        flux_y : (n_elems,) y yönü ısı akısı
        flux_mag : (n_elems,) toplam ısı akısı büyüklüğü
    """
    flux_x = []
    flux_y = []

    for elem_nodes in elements:
        coords = nodes[elem_nodes]
        x1,y1 = coords[0]
        x2,y2 = coords[1]
        x3,y3 = coords[2]
        A = 0.5 * abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))

        b = np.array([y2-y3, y3-y1, y1-y2])
        c = np.array([x3-x2, x1-x3, x2-x1])

        T_e = T[list(elem_nodes)]
        qx = -k * np.dot(b, T_e) / (2*A)
        qy = -k * np.dot(c, T_e) / (2*A)
        flux_x.append(qx)
        flux_y.append(qy)

    flux_x = np.array(flux_x)
    flux_y = np.array(flux_y)
    flux_mag = np.sqrt(flux_x**2 + flux_y**2)
    return flux_x, flux_y, flux_mag