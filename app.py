from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import numpy as np
import sys
sys.path.insert(0, '/workspaces/fem-solver')

from fem_solver.core.mesh import rectangular_mesh
from fem_solver.core.assembler import assemble
from fem_solver.core.solver import solve
from fem_solver.core.stress import compute_stress

app = FastAPI()

class AnalysisInput(BaseModel):
    width: float = 1.0
    height: float = 1.0
    nx: int = 10
    ny: int = 10
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
    nodes, elements = rectangular_mesh(data.width, data.height, data.nx, data.ny)
    
    K = assemble(nodes, elements, data.E, data.nu, data.t)
    
    f = np.zeros(2 * len(nodes))
    right_nodes = [i for i, n in enumerate(nodes) if abs(n[0] - data.width) < 1e-10]
    for n in right_nodes:
        f[2*n] = data.load

    fixed_dofs = []
    left_nodes = [i for i, n in enumerate(nodes) if abs(n[0]) < 1e-10]
    for n in left_nodes:
        fixed_dofs += [2*n, 2*n+1]

    u = solve(K, f, fixed_dofs)
    _, von_mises = compute_stress(nodes, elements, u, data.E, data.nu, data.t)

    return JSONResponse({
        "nodes": nodes.tolist(),
        "elements": elements.tolist(),
        "displacements": u.tolist(),
        "von_mises": von_mises.tolist(),
        "max_displacement_mm": float(np.max(np.abs(u)) * 1000),
        "max_von_mises_pa": float(von_mises.max())
    })