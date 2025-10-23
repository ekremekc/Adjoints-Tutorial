from dolfinx.fem import functionspace, Constant, Function, form, assemble_scalar, locate_dofs_topological, dirichletbc, Expression
from ufl import TrialFunction, dot, TestFunction, Measure, grad, as_vector, SpatialCoordinate, exp, inner
from dolfinx.fem.petsc import LinearProblem
from dolfinx import default_scalar_type
from dolfinx.io import XDMFFile
from dolfinx.io import gmshio
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np

kappa = 4
Q_total = 4  # W
u_edge = 50  # C

mesh, subdomains, facet_tags = gmshio.read_from_msh("MeshDir/3D_data.msh", MPI.COMM_WORLD, rank = 0, gdim = 3)

degree = 1

V = functionspace(mesh, ("Lagrange", degree))
u, p = TrialFunction(V), TestFunction(V)

# Define the boundary conditions

bcs = []
edge_tags = [13,14,15,17]
for tag in edge_tags:
    u_D = Function(V)
    u_D.x.array[:] = u_edge
    facets = facet_tags.find(tag)
    dofs = locate_dofs_topological(V, V.mesh.topology.dim - 1, facets)
    bcs.append(dirichletbc(u_D, dofs))

q_tag = 2
V2 = functionspace(mesh, ("DG", degree))
q = Function(V2)
dx = Measure("dx", subdomain_data=subdomains)
volume_form = form(Constant(mesh, PETSc.ScalarType(1)) * dx(q_tag))
V_pcb = MPI.COMM_WORLD.allreduce(assemble_scalar(volume_form), op=MPI.SUM)
q_tot = Q_total / V_pcb

subdomain_cells = subdomains.find(q_tag)
q.x.array[subdomain_cells] = np.full_like(subdomain_cells, q_tot, dtype=default_scalar_type)
q.x.scatter_forward()

dx = Measure("dx", domain=V.mesh, subdomain_data=subdomains)

kappa = default_scalar_type(kappa)
a_d = kappa * inner(grad(u), grad(p)) * dx
L_d = inner(q, p) * dx(q_tag)

# Solve direct problem
direct_problem = LinearProblem(a_d, L_d, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
u_direct = direct_problem.solve()

with XDMFFile(MPI.COMM_WORLD, "ResultsDir/u_direct.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_direct)

print("Max Temp: ", max(u_direct.x.array))

## Adjoint problem
x = SpatialCoordinate(mesh)
w0 = Function(V)
u_desired = 80
sigma = 0.005 # for demonstration

# Extract DoF values of u
imax = np.argmax(u_direct.x.array)
dof_coords = V.tabulate_dof_coordinates()
x_max = dof_coords[imax]
x_max_ufl = as_vector(x_max)
gaussian_expr = exp(-inner(x - x_max_ufl, x - x_max_ufl) / (2 * sigma**2))
C_sigma = mesh.comm.allreduce(assemble_scalar(form(gaussian_expr * dx)), op=MPI.SUM)
w_expr = gaussian_expr / C_sigma
w0.interpolate(Expression(w_expr, V.element.interpolation_points()))

with XDMFFile(MPI.COMM_WORLD, "ResultsDir/w_figure.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(w0)