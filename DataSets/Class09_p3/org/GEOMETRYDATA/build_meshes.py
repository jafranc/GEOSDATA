import numpy as np
import sys
import os
import meshio
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
#from scipy import interpolate

### 
##  pre-processing script using python vtk to build vtk mesh (Johansen_wip.vtk) 
##  from input data	and meshio to read and rewrite *normalized* msh and vtk.
##
##	(wip) experimental additional treatment to add boundary cells 
###

# constant unit conversion
mD = 1e-15
#infinity
nmax = 80000
cpath = os.getcwd()

# load geom
vert = np.loadtxt('./vertices_johansen.dat', skiprows=1)
elt = np.loadtxt('./elements_johansen.dat', skiprows=0, dtype=int)
elt = [ elt_ - 1 for elt_ in elt ] # C-indexing 


points_ = [ [pt[1], pt[2], pt[3]] for pt in vert ]
cells_ = [ [int(el[1]),\
            int(el[2]),\
            int(el[3]),\
            int(el[4]),\
            int(el[5]),\
            int(el[6]),\
            int(el[7]),\
            int(el[8])]\
            for el in elt[0:nmax] ]

# load properties
prop = np.loadtxt('properties_johansen.dat', skiprows=1)

mesh = meshio.Mesh(points_, [("hexahedron",cells_)] )
print( " Pre-writing mesh [VTK] ... ")
mesh.write('./Johansen_wip.vtk')

# re-reading mesh and ssigning fields
read = vtk.vtkUnstructuredGridReader()
read.SetFileName('./Johansen_wip.vtk')
read.Update()
ugrid = read.GetOutput()

from vtk.util.numpy_support import numpy_to_vtk
vtkporo = numpy_to_vtk( prop[:,3], 1 )
vtkporo.SetName('poro')
ugrid.GetPointData().AddArray( vtkporo )

#perm as a tensor
vtkperm = numpy_to_vtk( mD*np.array([prop[:,4],prop[:,4],prop[:,4]]).transpose() , 1 )
#perm as a scalar
#vtkperm = numpy_to_vtk( mD*prop[:,4] , 1 )
vtkperm.SetName('perm')
ugrid.GetPointData().AddArray( vtkperm )

algs_ = vtk.vtkPointDataToCellData()
algs_.SetInputData( ugrid )
algs_.Update()

ugrid = algs_.GetOutput()
ugrid.GetPointData().AddArray( vtkporo )
ugrid.GetPointData().AddArray( vtkperm )

#boundary_faces
print("tagging boundary faces ...")
bfaces = np.loadtxt('./boundary_faces_johansen.dat', skiprows=0, dtype=int)
kept = []

for faces in bfaces:
    nn = np.abs( np.cross( vert[faces[1]-1,1:] - vert[faces[2]-1,1:],\
            vert[faces[1]-1,1:] - vert[faces[3]-1,1:]) )
    nn = nn / np.linalg.norm(nn)
    if nn[2]<0.5:
        kept.append(faces[0])

kept = np.asarray(kept, dtype=int)

bfaces = np.unique( bfaces[kept-1,1:].ravel() )
bfaces = bfaces - 1 # C-indexing 
num_bpts = bfaces.shape[0]
#tagging boundary cells
tagged = []
for numel,el in enumerate(cells_):
        sdiff = np.setdiff1d( bfaces, np.asarray(el) )
        if ( sdiff.shape[0] < num_bpts ):
            tagged.append(numel)

np.savetxt('./bc_elements.dat', np.asarray(tagged), fmt='%i')

# reloading 
bcix = np.loadtxt('bc_elements.dat', dtype=int)
ncell = ugrid.GetNumberOfCells()

attributes = np.ones(ncell)
attributes[bcix] = 2
arr = dsa.numpy_support.numpy_to_vtk(attributes)
arr.SetName('attribute')
ugrid.GetCellData().AddArray(arr)
#re-write using vtk
writer = vtk.vtkUnstructuredGridWriter()
writer.SetFileName('./Johansen_wip.vtk')
writer.SetInputData( ugrid )
writer.Update()
writer.Write()

#re-read using meshio
mesh = meshio.read('./Johansen_wip.vtk') #file_format="vtk")
print( " Writing mesh [GMSH] ... ")
mesh.write('./Johansen.msh', file_format="gmsh22", binary=False )
print( " Writing mesh [VTK] ... ")
mesh.write('./Johansen.vtk', file_format="vtk", binary=False )
