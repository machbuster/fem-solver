# FEM Solver

Sıfırdan Python ile yazılmış 2D Finite Element Method solver. Web arayüzü ile tarayıcıdan çalışır.

## Özellikler
- CST (Constant Strain Triangle) eleman formülasyonu
- Otomatik dikdörtgen mesh üretici
- Global stiffness matrix assembly
- Dirichlet boundary condition (penalty method)
- Linear static solver
- Von Mises stress hesabı
- Web arayüzü (FastAPI + Plotly)

## Kullanılan Teknolojiler
- Python, NumPy, SciPy
- FastAPI (backend)
- Plotly (görselleştirme)

## Modüller
- `fem_solver/elements/cst.py` — CST eleman
- `fem_solver/core/assembler.py` — Global stiffness assembly
- `fem_solver/core/solver.py` — BC + linear solver
- `fem_solver/core/stress.py` — Von Mises stress
- `fem_solver/core/mesh.py` — Otomatik mesh üretici
- `app.py` — FastAPI web uygulaması

## Çalıştırmak için
pip install fastapi uvicorn numpy matplotlib

uvicorn app:app --host 0.0.0.0 --port 8000

## Sıradaki adımlar
- Q4 eleman desteği
- Gmsh ile CAD entegrasyonu
- Modal analiz
- Thermal analiz