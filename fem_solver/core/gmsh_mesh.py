import gmsh
import numpy as np

def _init_gmsh():
    try:
        gmsh.initialize()
    except Exception:
        pass
    gmsh.option.setNumber("General.Terminal", 0)

def rectangular_plate(width, height, mesh_size=0.1):
    _init_gmsh()
    gmsh.model.add("rect")
    gmsh.model.geo.addPoint(0, 0, 0, mesh_size, 1)
    gmsh.model.geo.addPoint(width, 0, 0, mesh_size, 2)
    gmsh.model.geo.addPoint(width, height, 0, mesh_size, 3)
    gmsh.model.geo.addPoint(0, height, 0, mesh_size, 4)
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    nodes, elements = _extract_mesh()
    gmsh.model.remove()
    return nodes, elements

def plate_with_hole(width, height, hole_radius, mesh_size=0.05):
    _init_gmsh()
    gmsh.model.add("hole")
    cx, cy = width/2, height/2
    gmsh.model.geo.addPoint(0, 0, 0, mesh_size, 1)
    gmsh.model.geo.addPoint(width, 0, 0, mesh_size, 2)
    gmsh.model.geo.addPoint(width, height, 0, mesh_size, 3)
    gmsh.model.geo.addPoint(0, height, 0, mesh_size, 4)
    gmsh.model.geo.addPoint(cx, cy, 0, mesh_size/2, 5)
    gmsh.model.geo.addPoint(cx+hole_radius, cy, 0, mesh_size/2, 6)
    gmsh.model.geo.addPoint(cx, cy+hole_radius, 0, mesh_size/2, 7)
    gmsh.model.geo.addPoint(cx-hole_radius, cy, 0, mesh_size/2, 8)
    gmsh.model.geo.addPoint(cx, cy-hole_radius, 0, mesh_size/2, 9)
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)
    gmsh.model.geo.addCircleArc(6, 5, 7, 5)
    gmsh.model.geo.addCircleArc(7, 5, 8, 6)
    gmsh.model.geo.addCircleArc(8, 5, 9, 7)
    gmsh.model.geo.addCircleArc(9, 5, 6, 8)
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
    gmsh.model.geo.addCurveLoop([5, 6, 7, 8], 2)
    gmsh.model.geo.addPlaneSurface([1, 2], 1)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    nodes, elements = _extract_mesh()
    gmsh.model.remove()
    return nodes, elements

def l_shaped_plate(size, thickness, mesh_size=0.05):
    _init_gmsh()
    gmsh.model.add("lshape")
    s, t = size, thickness
    pts = [(0,0),(s,0),(s,t),(t,t),(t,s),(0,s)]
    for i, (x, y) in enumerate(pts):
        gmsh.model.geo.addPoint(x, y, 0, mesh_size, i+1)
    for i in range(6):
        gmsh.model.geo.addLine(i+1, (i+1)%6+1, i+1)
    gmsh.model.geo.addCurveLoop([1,2,3,4,5,6], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    nodes, elements = _extract_mesh()
    gmsh.model.remove()
    return nodes, elements

def _extract_mesh():
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    nodes = np.array(coords).reshape(-1, 3)[:, :2]
    tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements()
    elements = []
    for etype, etags, enode_tags in zip(elem_types, elem_tags, elem_node_tags):
        if etype == 2:
            n = len(etags)
            conn = np.array(enode_tags).reshape(n, 3)
            for row in conn:
                elements.append([tag_to_idx[t] for t in row])
    return nodes, np.array(elements)