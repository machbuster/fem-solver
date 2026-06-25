# FEM Solver

2D Finite Element Method solver — sıfırdan Python ile yazıldı.

## Tamamlanan modüller
- `fem_solver/elements/cst.py` — CST (Constant Strain Triangle) eleman
- `fem_solver/core/assembler.py` — Global stiffness matrix assembly
- `fem_solver/core/solver.py` — Boundary condition + linear static solver

## Sıradaki adımlar
- Von Mises stress hesabı
- Matplotlib ile mesh + stress görselleştirme
- Web arayüzü

## Kullanılan yöntemler
- Plane stress formülasyonu
- Penalty method ile Dirichlet BC
- NumPy ile sparse olmayan direct solver

## Test
```
python fem_solver/tests/test_cst.py
```