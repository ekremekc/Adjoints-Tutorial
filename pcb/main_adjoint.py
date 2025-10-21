from dolfinx.fem import functionspace, Constant, Function, form, assemble_scalar, locate_dofs_topological, dirichletbc
from ufl import TrialFunction, dot, TestFunction, Measure, grad, max_value
from dolfinx.fem.petsc import LinearProblem
from dolfinx import default_scalar_type
from dolfinx.io import XDMFFile
from dolfinx.io import gmshio
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np

kappa = 4
Q_total = 10  # W
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
P = functionspace(mesh, ("DG", degree))
f = Function(P)
dx = Measure("dx", subdomain_data=subdomains)
volume_form = form(Constant(mesh, PETSc.ScalarType(1)) * dx(q_tag))
V_pcb = MPI.COMM_WORLD.allreduce(assemble_scalar(volume_form), op=MPI.SUM)
q_tot = Q_total / V_pcb

subdomain_cells = subdomains.find(q_tag)
f.x.array[subdomain_cells] = np.full_like(subdomain_cells, q_tot, dtype=default_scalar_type)
f.x.scatter_forward()

dx = Measure("dx", domain=V.mesh, subdomain_data=subdomains)

kappa = default_scalar_type(kappa)
a_d = kappa * dot(grad(u), grad(p)) * dx
L_d = dot(f, p) * dx(q_tag)

# Solve direct problem
direct_problem = LinearProblem(a_d, L_d, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
u_direct = direct_problem.solve()

with XDMFFile(MPI.COMM_WORLD, "ResultsDir/u_direct.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_direct)

# Adjoint problem
f_max = 80
u_max = u_direct.x.array.max()
print(u_max)
diff = default_scalar_type(u_max-f_max)
v, q = TrialFunction(V), TestFunction(V)
a_a = - kappa * dot(grad(v), grad(q)) * dx
L_a = dot(diff, p) * dx

# Solve direct problem
adjoint_problem = LinearProblem(a_a, L_a, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
u_adjoint = adjoint_problem.solve()

with XDMFFile(MPI.COMM_WORLD, "ResultsDir/u_adjoint.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_adjoint)