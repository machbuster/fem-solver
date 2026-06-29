import numpy as np
import gmsh

def dxf_to_mesh(dxf_path: str, mesh_size: float = 0.1):
    try:
        gmsh.initialize()
    except Exception:
        pass
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("dxf_model")

    # Gmsh DXF'i doğrudan okusun
    gmsh.merge(dxf_path)
    
    # Tüm curve'leri bul ve yüzey oluştur
    gmsh.model.geo.synchronize()
    
    # Curve loop otomatik bul
    curves = gmsh.model.getEntities(1)
    if not curves:
        raise ValueError("DXF dosyasında hiç çizgi bulunamadı.")
    
    curve_tags = [c[1] for c in curves]
    
    try:
        gmsh.model.geo.addCurveLoop(curve_tags, 1)
        gmsh.model.geo.addPlaneSurface([1], 1)
    except Exception:
        # Otomatik yüzey oluştur
        gmsh.model.mesh.classifySurfaces(np.pi, True, True, np.pi)
        gmsh.model.mesh.createGeometry()

    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.model.mesh.generate(2)

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

    gmsh.model.remove()
    return nodes, np.array(elements)