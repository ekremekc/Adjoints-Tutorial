
import numpy as np

import ufl
from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, grad, inner, dot

from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import pyvista as pv
from dolfinx.fem import Function, FunctionSpace 
# +
nx = 10
ny = 10

L_x = 1.0
L_y = 1.0 

msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
                            points=((0.0, 0.0), (L_x, L_y)), n=(nx, ny),
                            cell_type=mesh.CellType.triangle,)

V = fem.FunctionSpace(msh, ("Lagrange", 1))

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

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

x = ufl.SpatialCoordinate(msh)
f = fem.Constant(msh, ScalarType(5)) # -6

a = dot(grad(u), grad(v)) * dx
L = dot(f, v) * dx 

problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
u_direct = problem.solve()

# with io.XDMFFile(msh.comm, "out_poisson/poisson.xdmf", "w") as file:
#     file.write_mesh(msh)
#     file.write_function(u_direct)


# Adjoint problem

q = ufl.TrialFunction(V)
p = ufl.TestFunction(V)

bc_adjoint = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)


f_adjoint = fem.Function(V)
f_adjoint.x.array[:] = u_direct.x.array[:] - u_analytical.x.array[:]

a_adjoint = - dot(grad(q), grad(p)) * dx
L_adjoint = dot(f_adjoint, p) * dx 


problem_adjoint = fem.petsc.LinearProblem(a_adjoint, L_adjoint, bcs=[bc_adjoint], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
u_adjoint = problem_adjoint.solve()

# with io.XDMFFile(msh.comm, "out_poisson/poisson_adjoint.xdmf", "w") as file:
#     file.write_mesh(msh)
#     file.write_function(u_adjoint)

# Gradient of functional (minimizes the {u_direct - u_analytical}) with respect to f


alpha = 1e2


scalar_f = [5]
iterations = np.arange(0,51,1)


for i in range(50):

    
    u_direct = problem.solve()
    
    
    f_adjoint = fem.Function(V)
    f_adjoint.x.array[:] = u_direct.x.array[:] - u_analytical.x.array[:]

    a_adjoint = - dot(grad(q), grad(p)) * dx
    L_adjoint = dot(f_adjoint, p) * dx 


    problem_adjoint = fem.petsc.LinearProblem(a_adjoint, L_adjoint, bcs=[bc_adjoint], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    u_adjoint = problem_adjoint.solve()

    formm = fem.form(u_adjoint * dx)
    delJ_delf = fem.assemble_scalar(formm)
    # print("Derivative is :", delJ_delf)
    f.value = delJ_delf * alpha + f.value
    print("New f is: ", f.value)
    scalar_f.append(delJ_delf * alpha + f.value)

import matplotlib.pyplot as plt

plt.plot(iterations, scalar_f)
plt.xlabel("Iteration")
plt.ylabel("$f$")
plt.savefig("iterations.pdf")
plt.show()

with io.XDMFFile(msh.comm, "out_poisson/poisson.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(u_direct)