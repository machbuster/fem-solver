# FEM Solver

Sıfırdan Python ile yazılmış, web tabanlı 2D Sonlu Elemanlar Yöntemi (FEM) çözücüsü.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![Gmsh](https://img.shields.io/badge/Gmsh-4.x-orange)

## Özellikler

- CST (Constant Strain Triangle) eleman formülasyonu
- Otomatik mesh üretimi (Gmsh entegrasyonu)
- 3 farklı geometri: Dikdörtgen plaka, delikli plaka, L profil
- Kenar bazlı yük ve sınır koşulu seçimi (Sol/Sağ/Üst/Alt)
- 4 yük yönü: +X, -X, +Y, -Y
- Malzeme kütüphanesi: Çelik, Alüminyum, Titanyum, Karbon Fiber
- Von Mises stress hesabı ve güvenlik faktörü
- Interaktif web arayüzü (Plotly görselleştirme)
- PDF rapor çıktısı

## Kullanılan Teknolojiler

| Katman | Teknoloji |
|--------|-----------|
| FEM Çekirdeği | Python, NumPy, SciPy |
| Mesh | Gmsh |
| Backend | FastAPI |
| Frontend | HTML, JavaScript, Plotly |
| Rapor | ReportLab |

## Proje Yapısı

fem-solver/

├── fem_solver/

│   ├── elements/

│   │   └── cst.py          # CST eleman formülasyonu

│   ├── core/

│   │   ├── assembler.py    # Global stiffness matrix

│   │   ├── solver.py       # BC + linear solver

│   │   ├── stress.py       # Von Mises stress

│   │   ├── mesh.py         # Dikdörtgen mesh üretici

│   │   └── gmsh_mesh.py    # Gmsh entegrasyonu

│   └── tests/

│       └── test_cst.py

├── app.py                  # FastAPI uygulaması

└── index.html              # Web arayüzü

## Kurulum ve Çalıştırma

```bash
pip install fastapi uvicorn numpy scipy gmsh matplotlib reportlab
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Teorik Altyapı

- Plane stress formülasyonu
- CST eleman stiffness matrisi: `Ke = t * A * B^T * D * B`
- Dirichlet BC: Elimination method
- Von Mises: `σ_vm = sqrt(σx² - σx·σy + σy² + 3·τxy²)`

## Sıradaki Adımlar

- DXF dosyası import (CATIA/AutoCAD entegrasyonu)
- Q4 quadrilateral eleman
- Modal analiz (eigenvalue)
- Thermal analiz