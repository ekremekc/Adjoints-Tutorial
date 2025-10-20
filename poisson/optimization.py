from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from ufl import dx, grad, dot, TrialFunction, TestFunction, SpatialCoordinate
from petsc4py.PETSc import ScalarType
from mpi4py import MPI
import numpy as np
import pyvista as pv

nx, ny = 10, 10
L_x, L_y = 1.0, 1.0

msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
                            points=((0.0, 0.0), (L_x, L_y)), n=(nx, ny),
                            cell_type=mesh.CellType.triangle,)

V = fem.functionspace(msh, ("Lagrange", 1))

u_analytical = fem.Function(V)
u_analytical.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)

facets = mesh.locate_entities_boundary(msh, dim=1,
                                       marker=lambda x: np.logical_or(np.logical_or(np.isclose(x[0], 0.0),
                                                                                    np.isclose(x[0], L_x)),
                                                                      np.logical_or(np.isclose(x[1], 0.0),
                                                                                    np.isclose(x[1], L_y))))


dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)

# bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)
bc = fem.dirichletbc(u_analytical, dofs)

# Direct problem

u = TrialFunction(V)
v = TestFunction(V)

x = SpatialCoordinate(msh)
f = fem.Constant(msh, ScalarType(5)) # -6

a = dot(grad(u), grad(v)) * dx
L = dot(f, v) * dx 

problem_direct = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
u_direct = problem_direct.solve()

# Adjoint problem

q = TrialFunction(V)
p = TestFunction(V)

bc_adjoint = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

a_adjoint = - dot(grad(q), grad(p)) * dx
L_adjoint = dot(u_direct-u_analytical, p) * dx 

problem_adjoint = LinearProblem(a_adjoint, L_adjoint, bcs=[bc_adjoint], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
u_adjoint = problem_adjoint.solve()


dJ_df_form = fem.form(u_adjoint * dx)

alpha = 1e2

scalar_f = [5]
iterations = np.arange(0,51,1)

for i in range(50):
    
    u_direct = problem_direct.solve()
    u_adjoint = problem_adjoint.solve()

    delJ_delf = fem.assemble_scalar(dJ_df_form)
    print("gradient: ", delJ_delf)
    f.value = delJ_delf * alpha + f.value
    print("New f is: ", f.value)
    scalar_f.append(delJ_delf * alpha + f.value)

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
plt.plot(iterations, scalar_f, 'r-')
plt.xlabel("Iteration")
plt.ylabel("$f$")
plt.grid()
if alpha==1E2:
    plt.savefig("ResultsDir/Figure2b.png", dpi=300)
elif alpha==1E3:
    plt.savefig("ResultsDir/Figure2a.png", dpi=300)
plt.show()

with io.XDMFFile(msh.comm, "ResultsDir/u_opt.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(u_direct)