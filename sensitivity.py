import dolfinx
from dolfinx.fem import Function, FunctionSpace, Constant
from dolfinx.fem.function import VectorFunctionSpace
from dolfinx.mesh import MeshTags,locate_entities
from dolfinx.generation import UnitSquareMesh
from dolfinx.fem.assemble import assemble_matrix,assemble_scalar
from mpi4py import MPI
from ufl import Measure, FacetNormal, TestFunction, TrialFunction, dx, grad, inner, Dn, dot
from petsc4py import PETSc
import numpy as np
from slepc4py import SLEPc

# Define Mesh
mesh = UnitSquareMesh(MPI.COMM_WORLD, 8, 8, dolfinx.cpp.mesh.CellType.quadrilateral)

# Define Boundaries
boundaries = [(1, lambda x: np.isclose(x[0], 0)),
              (2, lambda x: np.isclose(x[0], 1)),
              (3, lambda x: np.isclose(x[1], 0)),
              (4, lambda x: np.isclose(x[1], 1))]

# Define Boundary Tags
facet_indices, facet_markers = [], []
fdim = mesh.topology.dim - 1
for (marker, locator) in boundaries:
    facets = locate_entities(mesh, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full(len(facets), marker))
facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = MeshTags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

V = FunctionSpace(mesh, ("Lagrange", 1))

u = TrialFunction(V)
v = TestFunction(V)

a = inner(grad(u), grad(v))*dx
A = assemble_matrix(a)
A.assemble()

c = -inner(u , v) * dx
C = assemble_matrix(c)
C.assemble()

solver = SLEPc.EPS().create(MPI.COMM_WORLD)
C = - C

solver.setOperators(A, C)
solver.setTwoSided(True)
solver.setFromOptions()
solver.solve()

A = solver.getOperators()[0]
vr, vi = A.createVecs()

eig = solver.getEigenvalue(0)
omega = np.sqrt(eig)
print(omega)

#DIRECT EIGENVECTOR
solver.getEigenvector(0, vr, vi)
p_dir = Function(V)
p_dir.vector.setArray(vr.array)

#ADJOINT EIGENVECTOR
solver.getLeftEigenvector(0, vr, vi)
p_adj = Function(V)
p_adj.vector.setArray(vr.array)

# SHAPE DERIVATIVES

def conjugate_function(p):
    p_conj = p
    p_conj.x.array[:] = np.conjugate(p_conj.x.array)
    return p_conj

ds = Measure("ds", domain=mesh, subdomain_data=facet_tag)

shape_gradient_form = ( Dn(conjugate_function(p_adj)) * Dn (p_dir))

C = Constant(mesh, PETSc.ScalarType(1))
A = assemble_scalar(C * ds(1))
C = 1/A # local average of shape derivative on edge 1

n = FacetNormal(mesh)
# Shape derivative of left edge (edge 1) with respect to eigenvalue

local_derivative = assemble_scalar(A * inner( shape_gradient_form, n) * ds(1))

print(local_derivative)

P = FunctionSpace(mesh, ("CG", 1))
der = Function(P)

# der.interpolate_cells(A*shape_gradient_form)