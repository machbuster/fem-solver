from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import numpy as np
import sys
sys.path.insert(0, '/workspaces/fem-solver')

from fem_solver.core.mesh import rectangular_mesh
from fem_solver.core.gmsh_mesh import rectangular_plate, plate_with_hole, l_shaped_plate
from fem_solver.core.assembler import assemble
from fem_solver.core.solver import solve
from fem_solver.core.stress import compute_stress

app = FastAPI()

class AnalysisInput(BaseModel):
    geometry: str = "rectangular"  # rectangular, hole, l_shape
    width: float = 1.0
    height: float = 1.0
    hole_radius: float = 0.2
    l_thickness: float = 0.3
    mesh_size: float = 0.1
    E: float = 210e9
    nu: float = 0.3
    t: float = 0.01
    load: float = 1000.0

@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.html") as f:
        return f.read()

@app.post("/analyze")
def analyze(data: AnalysisInput):
    # Mesh üret
    if data.geometry == "hole":
        nodes, elements = plate_with_hole(data.width, data.height, data.hole_radius, data.mesh_size)
    elif data.geometry == "l_shape":
        nodes, elements = l_shaped_plate(data.width, data.l_thickness, data.mesh_size)
    else:
        nodes, elements = rectangular_plate(data.width, data.height, data.mesh_size)

    K = assemble(nodes, elements, data.E, data.nu, data.t)

    f = np.zeros(2 * len(nodes))
    x_max = float(nodes[:,0].max())
    x_min = float(nodes[:,0].min())
    tol = (x_max - x_min) * 0.01

    # Sol kenar sabit
    fixed_dofs = []
    left_nodes = [i for i, n in enumerate(nodes) if n[0] < x_min + tol]
    for n in left_nodes:
        fixed_dofs += [2*n, 2*n+1]

    # Sağ kenara yük — eşit dağıt
    right_nodes = [i for i, n in enumerate(nodes) if n[0] > x_max - tol]
    load_per_node = data.load / len(right_nodes)
    for n in right_nodes:
        f[2*n] = load_per_node

    u = solve(K, f, fixed_dofs)
    _, von_mises = compute_stress(nodes, elements, u, data.E, data.nu, data.t)

    # nan kontrolü
    u = np.nan_to_num(u, nan=0.0)
    von_mises = np.nan_to_num(von_mises, nan=0.0)

    return JSONResponse({
        "nodes": nodes.tolist(),
        "elements": elements.tolist(),
        "displacements": u.tolist(),
        "von_mises": von_mises.tolist(),
        "max_displacement_mm": float(np.max(np.abs(u)) * 1000),
        "max_von_mises_pa": float(von_mises.max())
    })