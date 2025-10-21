import gmsh
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

filename = "3D_data"

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.option.setNumber("General.NumThreads", 8)

gmsh.model.add(filename)
gmsh.option.setString("Geometry.OCCTargetUnit", "M")

path = os.path.dirname(os.path.abspath(__file__))

gmsh.model.occ.importShapes(os.path.join(path, "GeomDir/" + filename + ".stp"))
gmsh.model.occ.removeAllDuplicates()
gmsh.model.occ.synchronize()

lc = 0.0080

led_tag = 2

# Mesh refinement
gmsh.model.mesh.field.add("Constant", 1)
gmsh.model.mesh.field.setNumbers(1, "VolumesList", [led_tag])
gmsh.model.mesh.field.setNumber(1, "VIn", lc / 10)
gmsh.model.mesh.field.setNumber(1, "VOut", lc)

gmsh.model.mesh.field.setAsBackgroundMesh(1)

gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
gmsh.option.setNumber("Mesh.Algorithm", 6)
gmsh.option.setNumber("Mesh.Algorithm3D", 10)
gmsh.option.setNumber("Mesh.Optimize", 1)
gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
gmsh.model.mesh.generate(3)

sur_tags = gmsh.model.getEntities(dim=2)

vol_tags = gmsh.model.getEntities(dim=3)


for surface in sur_tags:
    gmsh.model.addPhysicalGroup(2, [surface[1]], tag=surface[1])

for volume in vol_tags:
    gmsh.model.addPhysicalGroup(3, [volume[1]], tag=volume[1])

gmsh.model.occ.synchronize()

gmsh.write("{}.msh".format(dir_path + "/MeshDir/" + filename))
gmsh.write("{}.stl".format(dir_path + "/MeshDir/" + filename))

if "-nopopup" not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()