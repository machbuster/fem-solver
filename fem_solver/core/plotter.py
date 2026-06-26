import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def plot_results(nodes, elements, u, von_mises, scale=100):
    """
    3 panel: orijinal mesh, deforme mesh, Von Mises stress.
    
    scale : deformasyon büyütme faktörü (görsel için)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Deforme olmuş node koordinatları
    nodes_def = nodes.copy()
    for i in range(len(nodes)):
        nodes_def[i, 0] += scale * u[2*i]
        nodes_def[i, 1] += scale * u[2*i+1]

    # --- Panel 1: Orijinal Mesh ---
    ax = axes[0]
    for elem in elements:
        triangle = plt.Polygon(nodes[elem], fill=False, edgecolor='steelblue', linewidth=1.5)
        ax.add_patch(triangle)
    ax.scatter(nodes[:,0], nodes[:,1], color='steelblue', zorder=5, s=30)
    ax.set_title('Orijinal Mesh', fontsize=13)
    ax.set_aspect('equal')
    ax.autoscale()
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    # --- Panel 2: Deforme Mesh ---
    ax = axes[1]
    for elem in elements:
        triangle = plt.Polygon(nodes_def[elem], fill=False, edgecolor='coral', linewidth=1.5)
        ax.add_patch(triangle)
    ax.scatter(nodes_def[:,0], nodes_def[:,1], color='coral', zorder=5, s=30)
    ax.set_title(f'Deforme Mesh (x{scale})', fontsize=13)
    ax.set_aspect('equal')
    ax.autoscale()
    ax.set_xlabel('x (m)')

    # --- Panel 3: Von Mises Stress ---
    ax = axes[2]
    triang = tri.Triangulation(nodes[:,0], nodes[:,1], elements)
    
    # Her eleman için ortalama stress → node'lara dağıt
    node_vm = np.zeros(len(nodes))
    node_count = np.zeros(len(nodes))
    for i, elem in enumerate(elements):
        for n in elem:
            node_vm[n] += von_mises[i]
            node_count[n] += 1
    node_vm /= node_count

    tcf = ax.tricontourf(triang, node_vm, levels=14, cmap='jet')
    ax.triplot(triang, color='white', linewidth=0.5, alpha=0.4)
    plt.colorbar(tcf, ax=ax, label='Von Mises Stress (Pa)')
    ax.set_title('Von Mises Stress', fontsize=13)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')

    plt.suptitle('2D FEM Analiz Sonuçları', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fem_results.png', dpi=150, bbox_inches='tight')
    print("Görsel kaydedildi: fem_results.png")
    plt.show()