from time import time
import numpy as np
import random
from bsplinegenerator.bsplines import BsplineEvaluation
import matplotlib.pyplot as plt

### Control Points ###
# control_points = np.array([-3,-4,-2,-.5,1,0,2,3.5,3,6,8]) # 1 D
# control_points = np.array([[-3,-4,-2,-.5,1,0,2,3.5,3,5,6.5],
#                            [.5,3.5,6,5.5,3.7,2,-1,2,5,5.5,5]]) # 2 D
# control_points = np.array([[-3,  -4, -2, -.5, 1  ,   0,  2, 3.5, 3],
#                            [.5, 3.5,  6, 5.5, 3.7,   2, -1,   2, 5],
#                            [ 1, 3.2,  5,   0, 3.3, 1.5, -1, 2.5, 4]]) # 3D

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
    control_points = np.zeros((dimension,num_control_points))
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

order = 5
num_control_points = 11
dimension = random.randint(1, 3)

control_points = np.random.randint(10, size=(dimension,num_control_points)) # random
# control_points  = create_random_control_points_greater_than_angles(num_control_points,order,1,dimension)

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
spline_derivative_data, time_data = bspline.get_spline_derivative_data(num_data_points_per_interval,derivative_order)
spline_derivative_magnitude_data, time_data = bspline.get_derivative_magnitude_data(num_data_points_per_interval,derivative_order)
spline_curvature_data, time_data = bspline.get_spline_curvature_data(num_data_points_per_interval)
angular_rate_data, time_data = bspline.get_angular_rate_data(num_data_points_per_interval)
centripetal_acceleration_data, time_data = bspline.get_centripetal_acceleration_data(num_data_points_per_interval)
basis_function_data, time_data = bspline.get_basis_function_data(num_data_points_per_interval)
knot_points = bspline.get_knot_points()
defined_knot_points = bspline.get_defined_knot_points()
spline_at_knot_points = bspline.get_spline_at_knot_points()
bezier_control_points = bspline.get_bezier_control_points()
print("bspline_control_points:")
print(control_points)
print("bezier_control_points:")
print(bezier_control_points)
print("order: " , order)
print("control points: " , control_points)
print("knot_points: " , knot_points)
print("defined knot points: " , defined_knot_points)
print("spline at knots: " , spline_at_knot_points)
print("bezier_control_points: " , np.round(bezier_control_points,1))
print("max_derivative_magnitude: " , np.max(spline_derivative_magnitude_data))
print("max_curvature: " , np.max(spline_curvature_data))
print("max_angular_rate: " , np.max(angular_rate_data))
print("max_centripetal_acceleration: " , np.max(centripetal_acceleration_data))
print("number_of_basis_functions: " , len(basis_function_data))



##### Plot Spline Data
bspline.plot_spline(num_data_points_per_interval)
bspline.plot_spline_vs_time(num_data_points_per_interval)
bspline.plot_basis_functions(num_data_points_per_interval)
bspline.plot_derivative(num_data_points_per_interval, derivative_order)
bspline.plot_derivative_vs_time(num_data_points_per_interval, derivative_order)
bspline.plot_derivative_magnitude(num_data_points_per_interval, derivative_order)
bspline.plot_curvature(num_data_points_per_interval)
bspline.plot_angular_rate(num_data_points_per_interval)
bspline.plot_centripetal_acceleration(num_data_points_per_interval)
bspline.plot_bezier_curves(num_data_points_per_interval)
bspline.plot_minvo_curves(num_data_points_per_interval)

