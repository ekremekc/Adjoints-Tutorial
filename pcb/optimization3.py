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
Q_total = 1  # W
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

print("q_tot: ", q_tot)

subdomain_cells = subdomains.find(q_tag)
f.x.array[subdomain_cells] = np.full_like(subdomain_cells, q_tot, dtype=default_scalar_type)
f.x.scatter_forward()

dx = Measure("dx", domain=V.mesh, subdomain_data=subdomains)

kappa = default_scalar_type(kappa)
a_d = kappa * dot(grad(u), grad(p)) * dx
L_d = dot(f, p) * dx(q_tag)

# Solve direct problem
problem_direct = LinearProblem(a_d, L_d, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
u_direct = problem_direct.solve()

with XDMFFile(MPI.COMM_WORLD, "ResultsDir/u_direct.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_direct)

## Adjoint problem
# Get the coordinates of the corresponding DoFs
dof_coords = V.tabulate_dof_coordinates()

x = SpatialCoordinate(mesh)
w0 = Function(V)
u_desired = 80
sigma = 0.05

# Extract DoF values of u
imax = np.argmax(u_direct.x.array)
# u_max = Constant(V.mesh, default_scalar_type(u_direct.x.array[imax]))
x_max = dof_coords[imax]
x_max_ufl = as_vector(x_max)
gaussian_expr = exp(-inner(x - x_max_ufl, x - x_max_ufl) / (2 * sigma**2))
C_sigma = mesh.comm.allreduce(assemble_scalar(form(gaussian_expr * dx)), op=MPI.SUM)
w_expr = gaussian_expr / C_sigma
w0.interpolate(Expression(w_expr, V.element.interpolation_points()))

v, q = TrialFunction(V), TestFunction(V)
a_a = - kappa * dot(grad(v), grad(q)) * dx
L_a = dot(inner(w0,u_direct-u_desired), p) * dx
# L_a = dot(w0, p) * dx

bcs_adjoint = []
edge_tags = [13,14,15,17]
for tag in edge_tags:
    u_D = Function(V)
    u_D.x.array[:] = 0
    facets = facet_tags.find(tag)
    dofs = locate_dofs_topological(V, V.mesh.topology.dim - 1, facets)
    bcs_adjoint.append(dirichletbc(u_D, dofs))

# Solve adjoint problem
problem_adjoint = LinearProblem(a_a, L_a, bcs=bcs_adjoint, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
u_adjoint = problem_adjoint.solve()

with XDMFFile(MPI.COMM_WORLD, "ResultsDir/u_adjoint.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_adjoint)


dJ_df_form = form(u_adjoint * dx(q_tag))

alpha = 5e3

scalar_f = [Q_total]



for i in range(50):
    
    u_direct = problem_direct.solve()
    u_adjoint = problem_adjoint.solve()

    J = abs(u_direct.x.array.max()-u_desired)
    dJ_df = assemble_scalar(dJ_df_form)
    df_dQ = 1/V_pcb
    dJ_dQ = dJ_df*df_dQ
    print(f"Functional: {J:.3f}, delJ_delf: {dJ_df:.6f}, Q_tot: {q_tot*V_pcb:.3f} ")
    if J<2:
        break

    q_tot += alpha * dJ_dQ
    f.x.array[subdomain_cells] = np.full_like(subdomain_cells, q_tot, dtype=default_scalar_type)
    f.x.scatter_forward()
    Q_total=q_tot*V_pcb
    scalar_f.append(Q_total)


import matplotlib.pyplot as plt

# Set global font sizes for better consistency
plt.rcParams.update({'font.size': 18,          # Overall font size
                     'axes.titlesize': 16,     # Title size
                     'axes.labelsize': 16,     # Label size
                     'xtick.labelsize': 12,    # X-tick size
                     'ytick.labelsize': 12,    # Y-tick size
                     'legend.fontsize': 12,    # Legend size
                     })

fig, ax = plt.subplots(figsize=(8, 6)) # Use a specific size for better presentation
plt.plot(np.arange(0,len(scalar_f),1), scalar_f, 'r-')
plt.xlabel("Iteration")
plt.ylabel("$f$")
plt.grid()
if alpha==1E2:
    plt.savefig("ResultsDir/Figure2b.png", dpi=300)
elif alpha==1E3:
    plt.savefig("ResultsDir/Figure2a.png", dpi=300)
plt.show()