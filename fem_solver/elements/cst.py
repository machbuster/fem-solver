import numpy as np

class CSTElement:
    """
    Constant Strain Triangle (CST) element for 2D plane stress analysis.
    Nodes: 3 nodes, each with (x, y) coordinates.
    DOF: 6 (u1,v1, u2,v2, u3,v3)
    """

    def __init__(self, nodes: np.ndarray, E: float, nu: float, t: float = 1.0):
        """
        nodes : (3,2) array — [[x1,y1],[x2,y2],[x3,y3]]
        E     : Young's modulus
        nu    : Poisson's ratio
        t     : thickness
        """
        self.nodes = nodes
        self.E = E
        self.nu = nu
        self.t = t

    def area(self) -> float:
        x1,y1 = self.nodes[0]
        x2,y2 = self.nodes[1]
        x3,y3 = self.nodes[2]
        return 0.5 * abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))

    def compute_D(self) -> np.ndarray:
        """Plane stress constitutive matrix (3x3)."""
        E, nu = self.E, self.nu
        c = E / (1 - nu**2)
        return c * np.array([
            [1,  nu, 0],
            [nu, 1,  0],
            [0,  0,  (1-nu)/2]
        ])

    def compute_B(self) -> np.ndarray:
        """Strain-displacement matrix (3x6)."""
        x1,y1 = self.nodes[0]
        x2,y2 = self.nodes[1]
        x3,y3 = self.nodes[2]
        A = self.area()
        b1 = y2 - y3;  b2 = y3 - y1;  b3 = y1 - y2
        c1 = x3 - x2;  c2 = x1 - x3;  c3 = x2 - x1
        return (1/(2*A)) * np.array([
            [b1, 0,  b2, 0,  b3, 0 ],
            [0,  c1, 0,  c2, 0,  c3],
            [c1, b1, c2, b2, c3, b3]
        ])

    def compute_Ke(self) -> np.ndarray:
        """Element stiffness matrix (6x6). Ke = t * A * B.T @ D @ B"""
        B = self.compute_B()
        D = self.compute_D()
        A = self.area()
        return self.t * A * B.T @ D @ B