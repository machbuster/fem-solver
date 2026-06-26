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

MATERIALS = {
    "steel":      {"name": "Çelik (S235)",        "E": 210e9, "nu": 0.3,  "yield_mpa": 235},
    "aluminum":   {"name": "Alüminyum (Al6061)",   "E": 69e9,  "nu": 0.33, "yield_mpa": 276},
    "titanium":   {"name": "Titanyum (Ti-6Al-4V)", "E": 114e9, "nu": 0.34, "yield_mpa": 880},
    "carbon_fp":  {"name": "Karbon Fiber (CFRP)",  "E": 70e9,  "nu": 0.1,  "yield_mpa": 600},
    "custom":     {"name": "Özel",                 "E": 210e9, "nu": 0.3,  "yield_mpa": 250},
}

def get_materials():
    return {k: v["name"] for k, v in MATERIALS.items()}

app = FastAPI()

@app.get("/materials")


class AnalysisInput(BaseModel):
    geometry: str = "rectangular"  # rectangular, hole, l_shape
    material: str = "steel"
    width: float = 1.0
    height: float = 1.0
    hole_radius: float = 0.2
    l_thickness: float = 0.3
    mesh_size: float = 0.1
    E: float = 210e9
    nu: float = 0.3
    t: float = 0.01
    load: float = 1000.0
    load_direction: str = "x"      # x, y, -x, -y
    load_edge: str = "right"       # left, right, top, bottom
    fixed_edge: str = "left"       # left, right, top, bottom

@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.html") as f:
        return f.read()

@app.post("/analyze")
def analyze(data: AnalysisInput):
    mat = MATERIALS.get(data.material, MATERIALS["steel"])
    E = data.E if data.material == "custom" else mat["E"]
    nu = data.nu if data.material == "custom" else mat["nu"]
    yield_stress = mat["yield_mpa"] * 1e6

    # Mesh üret
    if data.geometry == "hole":
        nodes, elements = plate_with_hole(data.width, data.height, data.hole_radius, data.mesh_size)
    elif data.geometry == "l_shape":
        nodes, elements = l_shaped_plate(data.width, data.l_thickness, data.mesh_size)
    else:
        nodes, elements = rectangular_plate(data.width, data.height, data.mesh_size)

    K = assemble(nodes, elements, E, nu, data.t)
    
    f = np.zeros(2 * len(nodes))
    x_max = float(nodes[:,0].max())
    x_min = float(nodes[:,0].min())
    y_max = float(nodes[:,1].max())
    y_min = float(nodes[:,1].min())
    tol_x = (x_max - x_min) * 0.01
    tol_y = (y_max - y_min) * 0.01

    # Kenar seçimi
    def get_edge_nodes(edge):
        if edge == "left":
            return [i for i, n in enumerate(nodes) if n[0] < x_min + tol_x]
        elif edge == "right":
            return [i for i, n in enumerate(nodes) if n[0] > x_max - tol_x]
        elif edge == "bottom":
            return [i for i, n in enumerate(nodes) if n[1] < y_min + tol_y]
        elif edge == "top":
            return [i for i, n in enumerate(nodes) if n[1] > y_max - tol_y]
        return []

    # Sabit kenar
    fixed_dofs = []
    for n in get_edge_nodes(data.fixed_edge):
        fixed_dofs += [2*n, 2*n+1]

    # Yük yönü
    dir_map = {"x": 0, "-x": 0, "y": 1, "-y": 1}
    sign_map = {"x": 1, "-x": -1, "y": 1, "-y": -1}
    dof_offset = dir_map[data.load_direction]
    sign = sign_map[data.load_direction]

    load_nodes = get_edge_nodes(data.load_edge)
    load_per_node = sign * data.load / max(len(load_nodes), 1)
    for n in load_nodes:
        f[2*n + dof_offset] = load_per_node

    u = solve(K, f, fixed_dofs)
    _, von_mises = compute_stress(nodes, elements, u, E, nu, data.t)

    # nan kontrolü
    u = np.nan_to_num(u, nan=0.0)
    von_mises = np.nan_to_num(von_mises, nan=0.0)

    return JSONResponse({
        "nodes": nodes.tolist(),
        "elements": elements.tolist(),
        "displacements": u.tolist(),
        "von_mises": von_mises.tolist(),
        "max_displacement_mm": float(np.max(np.abs(u)) * 1000),
        "max_von_mises_pa": float(von_mises.max()),
        "yield_stress_mpa": float(yield_stress / 1e6),
        "safety_factor": float(yield_stress / von_mises.max()) if von_mises.max() > 0 else 999
    })