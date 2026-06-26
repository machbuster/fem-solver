import numpy as np

def rectangular_mesh(width, height, nx, ny):
    """
    Dikdörtgen plaka için otomatik CST mesh üretici.
    
    width  : plakanın genişliği (x yönü)
    height : plakanın yüksekliği (y yönü)
    nx     : x yönünde eleman sayısı
    ny     : y yönünde eleman sayısı
    
    Returns:
        nodes    : (n_nodes, 2)
        elements : (n_elems, 3) — her dikdörtgen 2 CST elemana bölünür
    """
    dx = width / nx
    dy = height / ny

    # Node koordinatları
    nodes = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            nodes.append([i * dx, j * dy])
    nodes = np.array(nodes)

    # Eleman bağlantıları
    elements = []
    for j in range(ny):
        for i in range(nx):
            # Her dikdörtgen → 2 üçgen
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n0 + (nx + 1)
            n3 = n2 + 1
            elements.append([n0, n1, n3])
            elements.append([n0, n3, n2])
    elements = np.array(elements)

    return nodes, elements