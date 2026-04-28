import torch


# base class for the electri fields

x_axis_size = 100
y_axis_size = 100
z_axis_size = 100


# initialize the starting state of the magnet fields

Ex = torch.zeros(x_axis_size, y_axis_size, z_axis_size)
Ey = torch.zeros(x_axis_size, y_axis_size, z_axis_size)
Ez = torch.zeros(x_axis_size, y_axis_size, z_axis_size)

Mx = torch.zeros(x_axis_size, y_axis_size, z_axis_size)
My = torch.zeros(x_axis_size, y_axis_size, z_axis_size)
Mz = torch.zeros(x_axis_size, y_axis_size, z_axis_size)


