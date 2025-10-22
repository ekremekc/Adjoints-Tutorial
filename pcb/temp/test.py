import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl

# --- 1. Create mesh and function space
domain = mesh.create_unit_square(MPI.COMM_WORLD, 64, 64)
V = fem.functionspace(domain, ("CG", 1))

# --- 2. Define parameters
x_max = (0.7, 0.3)  # Use plain tuple, not np.array
x_max_ufl = ufl.as_vector(x_max)

sigma = 0.05

# --- 3. Define Gaussian expression w̃(x) = exp(-|x - x_max|^2 / (2σ^2))
x = ufl.SpatialCoordinate(domain)
gaussian_expr = ufl.exp(-ufl.inner(x - x_max_ufl, x - x_max_ufl) / (2 * sigma**2))

# --- 4. Compute normalization constant Cσ = ∫Ω w̃(x) dx
C_sigma = fem.assemble_scalar(fem.form(gaussian_expr * ufl.dx(domain)))
C_sigma = domain.comm.allreduce(C_sigma, op=MPI.SUM)

# --- 5. Define normalized Gaussian function w₀(x)
w_expr = gaussian_expr / C_sigma

# --- 6. Interpolate onto the finite element space
w0 = fem.Function(V)
w0.interpolate(fem.Expression(w_expr, V.element.interpolation_points()))

# --- 7. Optional: print integral check
int_w0 = fem.assemble_scalar(fem.form(w0 * ufl.dx(domain)))
int_w0 = domain.comm.allreduce(int_w0, op=MPI.SUM)
if domain.comm.rank == 0:
    print(f"∫ w₀ dx = {int_w0:.6f} (should be 1.0)")

# w0 is now a smooth, normalized Gaussian weighting function
