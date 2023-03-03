from time import time
import numpy as np
import random
from bsplinegenerator.bsplines import BsplineEvaluation
import matplotlib.pyplot as plt

### Control Points ###

order = 5
dimension = 3
num_control_points = order + 1

def create_random_control_points_greater_than_angles(num_control_points,order,length,dimension):
    if order == 1:
        angle = np.pi/2
    elif order == 2:
        angle = np.pi/2
    elif order == 3:
        angle = np.pi/4
    elif order == 4:
        angle = np.pi/6
    elif order == 5:
        angle = np.pi/8
    control_points = np.zeros((dimension, num_control_points))
    for i in range(num_control_points):
        if i == 0:
            control_points[:,i][:,None] = np.array([[0],[0]])
        elif i == 1:
            random_vec = np.random.rand(2,1)
            next_vec = length*random_vec/np.linalg.norm(random_vec)
            control_points[:,i][:,None] = control_points[:,i-1][:,None] + next_vec
        else:
            new_angle = 2*np.pi*(0.5-np.random.rand())
            # new_angle = angle#*2*(0.5-np.random.rand())
            R = np.array([[np.cos(new_angle), -np.sin(new_angle)],[np.sin(new_angle), np.cos(new_angle)]])
            prev_vec = control_points[:,i-1][:,None] - control_points[:,i-2][:,None]
            unit_prev_vec = prev_vec/np.linalg.norm(prev_vec)
            next_vec = length*np.dot(R,unit_prev_vec)#*np.random.rand()
            control_points[:,i][:,None] = control_points[:,i-1][:,None] + next_vec
    return control_points

control_points = np.random.randint(100, size=(dimension,num_control_points)) # random
control_points  = create_random_control_points_greater_than_angles(num_control_points,order,1,dimension)


if len(control_points) == 1:
    control_points = control_points.flatten()

### Parameters
start_time = 0
scale_factor = 1
# derivative_order = random.randint(1, order)
derivative_order = 1
clamped = False
num_data_points_per_interval = 1000

### Create B-Spline Object ###
bspline = BsplineEvaluation(control_points, order, start_time, scale_factor, clamped)

#### Evaluate B-Spline Data ###
spline_data, time_data = bspline.get_spline_data(num_data_points_per_interval)
bezier_control_points = bspline.get_bezier_control_points()
minvo_control_points = bspline.get_minvo_control_points()
##### Plot Spline Data


# plt.plot(spline_data[0,:], spline_data[1,:], color="tab:blue")
# plt.scatter(control_points[0,:], control_points[1,:], color="tab:orange")
# plt.plot(control_points[0,:], control_points[1,:], color="tab:orange")
# plt.scatter(bezier_control_points[0,:], bezier_control_points[1,:], color="tab:red")
# plt.plot(bezier_control_points[0,:], bezier_control_points[1,:], color="tab:red")
# plt.scatter(minvo_control_points[0,:], minvo_control_points[1,:], color="tab:green")
# plt.plot(minvo_control_points[0,:], minvo_control_points[1,:], color="tab:green")
# plt.show()



