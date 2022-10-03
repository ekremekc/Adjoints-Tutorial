
import numpy as np

import ufl
from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, grad, inner

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

# +
nx = 32
ny = 32

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

bc = fem.dirichletbc(u_analytical, dofs)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

x = ufl.SpatialCoordinate(msh)
f = ScalarType(5) 

a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx 


problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# +
with io.XDMFFile(msh.comm, "out_poisson/poisson.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(uh)


# -
# +
try:
    import pyvista
    cells, types, x = plot.create_vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = uh.x.array.real
    grid.set_active_scalars("u")
    plotter = pyvista.Plotter()
    plotter.camera.position = (0.5, 0.50, 1.0)
    plotter.add_mesh(grid, show_edges=False)
    warped = grid.warp_by_scalar()
    #plotter.add_mesh(warped)
    #plotter.show(screenshot='direct.png')
except ModuleNotFoundError:
    print("'pyvista' is required to visualise the solution")
    print("Install 'pyvista' with pip: 'python3 -m pip install pyvista'")
# -
