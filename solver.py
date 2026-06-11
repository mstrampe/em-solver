import torch
import pyvista as pv
import numpy as np
import torch.nn.functional as F


# Base class for the electric and magnetic fields
class Field :
    """
    Base class that contains the electric and magnetic fields in a 3D grid.  This is a singular 
    state of fields.
    """
    def __init__(self, shape):
        
        self.shape = shape

        # Electric Field Vectors
        self.Ex = torch.zeros(shape[0], shape[1], shape[2])
        self.Ey = torch.zeros(shape[0], shape[1], shape[2])
        self.Ez = torch.zeros(shape[0], shape[1], shape[2])

        # Magnetic Field Vectors
        self.Hx = torch.zeros(shape[0], shape[1], shape[2])
        self.Hy = torch.zeros(shape[0], shape[1], shape[2])
        self.Hz = torch.zeros(shape[0], shape[1], shape[2])

        # Magnetization Vectors
        self.Mx = torch.zeros(shape[0], shape[1], shape[2])
        self.My = torch.zeros(shape[0], shape[1], shape[2])
        self.Mz = torch.zeros(shape[0], shape[1], shape[2])

        # conductivity
        self.sigma = torch.ones(shape[0], shape[1], shape[2])

        # Permittivity
        self.epsilon = torch.ones(shape[0], shape[1], shape[2])

        # Permeability
        self.mu = torch.ones(shape[0], shape[1], shape[2])



    def visualize_electric_field(self):
        ex = self.Ex.numpy()
        ey = self.Ey.numpy()
        ez = self.Ez.numpy()
        grid = pv.ImageData()
        grid.dimensions = np.array(self.shape)
        grid.origin = (0,0,0)
        grid.spacing = (1,1,1)
        vectors = np.column_stack((ex.flatten(), ey.flatten(), ez.flatten()))
        grid.point_data["Electric Field"] = vectors
        arrows = grid.glyph(orient="Electric Field", scale="Electric Field", factor=0.5)
        plotter = pv.Plotter()
        plotter.add_mesh(arrows, cmap="viridis")
        plotter.add_axes()
        plotter.show()
    
    def visualize_magnetic_field(self):
        mx = self.Mx.numpy()
        my = self.My.numpy()
        mz = self.Mz.numpy()
        grid = pv.ImageData()
        grid.dimensions = np.array(self.shape)
        grid.origin = (0,0,0)
        grid.spacing = (1,1,1)
        vectors = np.column_stack((mx.flatten(), my.flatten(), mz.flatten()))
        grid.point_data["Magnetic Field"] = vectors
        arrows = grid.glyph(orient="Magnetic Field", scale="Magnetic Field", factor=0.5)
        plotter = pv.Plotter()
        plotter.add_mesh(arrows, cmap="viridis")
        plotter.add_axes()
        plotter.show()

# base class for the electri fields



class FDTDSolver:
    """
    FDTD Solver algorithm.  this calss accepts an initial field configuration, and then 
    iterates over the fields using the FDTD algorithm.  It returns the final configuration.
    
    TODO: should contain a visualizers that allow one to see the field solving every step
    of the way
    """

    def __init__(self, steps = 100, interval = .01):
        self.steps = steps
        self.interval = interval

    def solve(self, field ):
        """
        solve will iterate over the time interval and solve for the final state of E / H.  
        this is the main FDTD algorithm of the Yee Lattice
        """

        for i in range(self.steps) :
            # update H
            field.Hx = field.Hx - (self.interval) / field.mu * (
                    (field.Ey - F.pad(field.Ey[:,:,:-1],(1,0,0,0,0,0), mode = 'constant', value = 0.0)) -
                    (field.Ez - F.pad(field.Ez[:,:-1,:],(0,0,1,0,0,0), mode = 'constant', value = 0.0)))  

            field.Hy = field.Hy - (self.interval) / field.mu * (
                    (field.Ez - F.pad(field.Ez[:-1,:,:],(0,0,0,0,1,0), mode = 'constant', value = 0.0)) -
                    (field.Ex - F.pad(field.Ex[:,:,:-1],(1,0,0,0,0,0), mode = 'constant', value = 0.0)))

            field.Hz = field.Hz - (self.interval) / field.mu * (
                    (field.Ex - F.pad(field.Ex[:,:-1,:],(0,0,1,0,0,0), mode = 'constant', value = 0.0)) -
                    (field.Ey - F.pad(field.Ey[:-1,:,:],(0,0,0,0,1,0), mode = 'constant', value = 0.0)))

            # update E
            field.Ex = field.Ex - (self.interval) * field.sigma / field.epsilon * field.Ex + (self.interval) / field.epsilon * (
                    (field.Hz - F.pad(field.Hz[:,:-1,:],(0,0,1,0,0,0), mode = 'constant', value = 0.0)) -
                    (field.Hy - F.pad(field.Hy[:,:,:-1],(1,0,0,0,0,0), mode = 'constant', value = 0.0)))

            field.Ey = field.Ey - (self.interval) * field.sigma / field.epsilon * field.Ey + (self.interval) / field.epsilon * (
                    (field.Hx - F.pad(field.Hx[:,:,:-1],(1,0,0,0,0,0), mode = 'constant', value = 0.0)) -
                    (field.Hz - F.pad(field.Hz[:-1,:,:],(0,0,0,0,1,0), mode = 'constant', value = 0.0)))

            field.Ez = field.Ez - (self.interval) * field.sigma / field.epsilon * field.Ez + (self.interval) / field.epsilon * (
                    (field.Hy - F.pad(field.Hy[:-1,:,:],(0,0,0,0,1,0), mode = 'constant', value = 0.0)) -
                    (field.Hx - F.pad(field.Hx[:,:-1,:],(0,0,1,0,0,0), mode = 'constant', value = 0.0)))

        return field


shape = (50,50,50)
field = Field(shape)



