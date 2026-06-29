from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import io
from pydantic import BaseModel
import numpy as np
from fem_solver.core.modal import compute_mass_matrix, modal_analysis
from fem_solver.core.thermal import compute_thermal_stiffness, solve_thermal, compute_heat_flux
import sys
sys.path.insert(0, '/workspaces/fem-solver')

from fem_solver.core.mesh import rectangular_mesh
from fem_solver.core.gmsh_mesh import rectangular_plate, plate_with_hole, l_shaped_plate
from fem_solver.core.assembler import assemble
from fem_solver.core.solver import solve
from fem_solver.core.stress import compute_stress

class AnalysisInput(BaseModel):
    geometry: str = "rectangular"
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
    load_direction: str = "x"
    load_edge: str = "right"
    fixed_edge: str = "left"
    n_modes: int = 6
    rho: float = 7850.0

MATERIALS = {
    "steel":      {"name": "Çelik (S235)",        "E": 210e9, "nu": 0.3,  "yield_mpa": 235},
    "aluminum":   {"name": "Alüminyum (Al6061)",   "E": 69e9,  "nu": 0.33, "yield_mpa": 276},
    "titanium":   {"name": "Titanyum (Ti-6Al-4V)", "E": 114e9, "nu": 0.34, "yield_mpa": 880},
    "carbon_fp":  {"name": "Karbon Fiber (CFRP)",  "E": 70e9,  "nu": 0.1,  "yield_mpa": 600},
    "custom":     {"name": "Özel",                 "E": 210e9, "nu": 0.3,  "yield_mpa": 250},
}

app = FastAPI()

@app.get("/materials")
def get_materials():
    return {k: v["name"] for k, v in MATERIALS.items()}

@app.post("/report")
@app.post("/report")
def generate_report(data: AnalysisInput):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    def tr(text):
        replacements = {
            'ş': 's', 'Ş': 'S', 'ı': 'i', 'İ': 'I',
            'ğ': 'g', 'Ğ': 'G', 'ü': 'u', 'Ü': 'U',
            'ö': 'o', 'Ö': 'O', 'ç': 'c', 'Ç': 'C'
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(tr("FEM Analiz Raporu"), styles['Title']))
    story.append(Spacer(1, 12))

    story.append(Paragraph(tr("Geometri"), styles['Heading2']))
    geo_data = [
        [tr("Parametre"), tr("Deger")],
        [tr("Geometri Tipi"), data.geometry],
        [tr("Genislik"), f"{data.width} m"],
        [tr("Yukseklik"), f"{data.height} m"],
        [tr("Kalinlik"), f"{data.t} m"],
        [tr("Mesh Boyutu"), f"{data.mesh_size} m"],
    ]
    geo_table = Table(geo_data, colWidths=[200, 200])
    geo_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e94560')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f5f5f5')),
    ]))
    story.append(geo_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph(tr("Malzeme"), styles['Heading2']))
    mat = MATERIALS.get(data.material, MATERIALS["steel"])
    mat_data = [
        [tr("Parametre"), tr("Deger")],
        [tr("Malzeme"), tr(mat["name"])],        
        ["Young Modulu", f"{mat['E']/1e9:.0f} GPa"],
        ["Poisson Orani", f"{mat['nu']}"],
        ["Akma Siniri", f"{mat['yield_mpa']} MPa"],
    ]
    mat_table = Table(mat_data, colWidths=[200, 200])
    mat_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e94560')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f5f5f5')),
    ]))
    story.append(mat_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph(tr("Yuk ve Sinir Kosullari"), styles['Heading2']))
    load_data = [
        [tr("Parametre"), tr("Deger")],
        ["Sabit Kenar", data.fixed_edge],
        ["Yuk Kenari", data.load_edge],
        ["Yuk Yonu", data.load_direction],
        ["Yuk Buyuklugu", f"{data.load} N"],
    ]
    load_table = Table(load_data, colWidths=[200, 200])
    load_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e94560')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f5f5f5')),
    ]))
    story.append(load_table)

    doc.build(story)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=fem_raporu.pdf"}
    )

@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.html") as f:
        return f.read()

MATERIAL_DENSITY = {
    "steel":     7850,
    "aluminum":  2700,
    "titanium":  4430,
    "carbon_fp": 1600,
    "custom":    7850,
}

MATERIAL_CONDUCTIVITY = {
    "steel":     50.0,
    "aluminum":  205.0,
    "titanium":  7.2,
    "carbon_fp": 5.0,
    "custom":    50.0,
}

class ThermalInput(BaseModel):
    geometry: str = "rectangular"
    material: str = "steel"
    width: float = 1.0
    height: float = 1.0
    hole_radius: float = 0.2
    l_thickness: float = 0.3
    mesh_size: float = 0.1
    t: float = 0.01
    k: float = 50.0
    T_hot: float = 100.0
    T_cold: float = 0.0
    hot_edge: str = "left"
    cold_edge: str = "right"

@app.post("/thermal")
def thermal(data: ThermalInput):
    if data.geometry == "hole":
        nodes, elements = plate_with_hole(data.width, data.height, data.hole_radius, data.mesh_size)
    elif data.geometry == "l_shape":
        nodes, elements = l_shaped_plate(data.width, data.l_thickness, data.mesh_size)
    else:
        nodes, elements = rectangular_plate(data.width, data.height, data.mesh_size)

    mat = MATERIALS.get(data.material, MATERIALS["steel"])
    k = MATERIAL_CONDUCTIVITY.get(data.material, 50.0)

    K_th = compute_thermal_stiffness(nodes, elements, k, data.t)
    T = solve_thermal(K_th, nodes, data.hot_edge, data.cold_edge, data.T_hot, data.T_cold)
    _, _, flux_mag = compute_heat_flux(nodes, elements, T, k)

    T = np.nan_to_num(T, nan=0.0)
    flux_mag = np.nan_to_num(flux_mag, nan=0.0)

    return JSONResponse({
        "nodes": nodes.tolist(),
        "elements": elements.tolist(),
        "temperature": T.tolist(),
        "heat_flux": flux_mag.tolist(),
        "T_max": float(T.max()),
        "T_min": float(T.min()),
        "flux_max": float(flux_mag.max()),
    })

@app.post("/modal")
def modal(data: AnalysisInput):
    if data.geometry == "hole":
        nodes, elements = plate_with_hole(data.width, data.height, data.hole_radius, data.mesh_size)
    elif data.geometry == "l_shape":
        nodes, elements = l_shaped_plate(data.width, data.l_thickness, data.mesh_size)
    else:
        nodes, elements = rectangular_plate(data.width, data.height, data.mesh_size)

    mat = MATERIALS.get(data.material, MATERIALS["steel"])
    E = data.E if data.material == "custom" else mat["E"]
    nu = data.nu if data.material == "custom" else mat["nu"]
    rho = data.rho if data.material == "custom" else MATERIAL_DENSITY.get(data.material, 7850)

    K = assemble(nodes, elements, E, nu, data.t)
    M = compute_mass_matrix(nodes, elements, rho, data.t)

    x_min = float(nodes[:,0].min())
    tol_x = (nodes[:,0].max() - x_min) * 0.01
    left_nodes = [i for i, n in enumerate(nodes) if n[0] < x_min + tol_x]
    fixed_dofs = []
    for n in left_nodes:
        fixed_dofs += [2*n, 2*n+1]

    freqs, mode_shapes = modal_analysis(K, M, fixed_dofs, n_modes=data.n_modes)

    return JSONResponse({
        "frequencies_hz": freqs.tolist(),
        "mode_shapes": mode_shapes.tolist(),
        "nodes": nodes.tolist(),
        "elements": elements.tolist(),
    })

@app.post("/analyze")
def analyze(data: AnalysisInput):
    if data.geometry == "hole":
        nodes, elements = plate_with_hole(data.width, data.height, data.hole_radius, data.mesh_size)
    elif data.geometry == "l_shape":
        nodes, elements = l_shaped_plate(data.width, data.l_thickness, data.mesh_size)
    else:
        nodes, elements = rectangular_plate(data.width, data.height, data.mesh_size)

    mat = MATERIALS.get(data.material, MATERIALS["steel"])
    E = data.E if data.material == "custom" else mat["E"]
    nu = data.nu if data.material == "custom" else mat["nu"]
    yield_stress = mat["yield_mpa"] * 1e6

    K = assemble(nodes, elements, E, nu, data.t)

    f = np.zeros(2 * len(nodes))
    x_max = float(nodes[:,0].max())
    x_min = float(nodes[:,0].min())
    y_max = float(nodes[:,1].max())
    y_min = float(nodes[:,1].min())
    tol_x = (x_max - x_min) * 0.01
    tol_y = (y_max - y_min) * 0.01

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

    fixed_dofs = []
    for n in get_edge_nodes(data.fixed_edge):
        fixed_dofs += [2*n, 2*n+1]

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
        "safety_factor": float(yield_stress / von_mises.max()) if von_mises.max() > 0 else 999,
    })