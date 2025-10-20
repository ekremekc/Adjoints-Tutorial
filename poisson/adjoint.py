from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from ufl import dx, grad, dot, TrialFunction, TestFunction, SpatialCoordinate
from petsc4py.PETSc import ScalarType
from mpi4py import MPI
import numpy as np
import pyvista as pv

nx, ny = 32, 32
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

bc = fem.dirichletbc(u_analytical, dofs)

# Direct problem

u = TrialFunction(V)
v = TestFunction(V)

x = SpatialCoordinate(msh)
f = fem.Constant(msh, ScalarType(5)) # -6

a = dot(grad(u), grad(v)) * dx
L = dot(f, v) * dx 

problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
u_direct = problem.solve()

with io.XDMFFile(msh.comm, "ResultsDir/u_direct.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(u_direct)

# Adjoint problem

q = TrialFunction(V)
p = TestFunction(V)

bc_adjoint = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

a_adjoint = - dot(grad(q), grad(p)) * dx
L_adjoint = dot(u_direct-u_analytical, p) * dx 


problem = LinearProblem(a_adjoint, L_adjoint, bcs=[bc_adjoint], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
u_adjoint = problem.solve()

with io.XDMFFile(msh.comm, "ResultsDir/u_adjoint.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(u_adjoint)

# Gradient of functional (minimizes the {u_direct - u_analytical}) with respect to f

formm = fem.form(u_adjoint * dx)
delJ_delf = fem.assemble_scalar(formm)
print("Derivative is :", delJ_delf)

import pyvista
msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim)
topology, cell_types, geometry = plot.vtk_mesh(msh, msh.topology.dim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

plotter = pv.Plotter(shape=(1, 3))

plotter.subplot(0, 0)
plotter.add_text("Direct", font_size=30)
grid1 = pyvista.UnstructuredGrid(topology, cell_types, geometry)
grid1.point_data["u"] = u_direct.x.array.real
plotter.add_mesh(grid1, show_edges=False)

plotter.subplot(0, 1)
plotter.add_text("Adjoint", font_size=30)
grid2 = pyvista.UnstructuredGrid(topology, cell_types, geometry)
grid2.point_data["u"] = u_adjoint.x.array.real
plotter.add_mesh(grid2, show_edges=False)

plotter.subplot(0, 2)
plotter.add_text("Analytical", font_size=30)
grid3 = pyvista.UnstructuredGrid(topology, cell_types, geometry)
grid3.point_data["u"] = u_analytical.x.array.real
plotter.add_mesh(grid3, show_edges=False)

plotter.add_axes(interactive=True)
plotter.show()