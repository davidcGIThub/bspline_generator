from time import time
import numpy as np
import random
from bsplinegenerator.bsplines import BsplineEvaluation
import matplotlib.pyplot as plt

### Control Points ###

order = 7
dimension = 2
num_control_points = order + 1

control_points = np.random.randint(100, size=(dimension,num_control_points)) # random
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
bezier_control_points = bspline.get_bezier_control_points()
minvo_control_points = bspline.get_minvo_control_points()
##### Plot Spline Data


plt.plot(spline_data[0,:], spline_data[1,:], color="tab:blue")
# plt.scatter(control_points[0,:], control_points[1,:], color="tab:orange")
# plt.plot(control_points[0,:], control_points[1,:], color="tab:orange")
plt.scatter(bezier_control_points[0,:], bezier_control_points[1,:], color="tab:red")
plt.plot(bezier_control_points[0,:], bezier_control_points[1,:], color="tab:red")
plt.scatter(minvo_control_points[0,:], minvo_control_points[1,:], color="tab:green")
plt.plot(minvo_control_points[0,:], minvo_control_points[1,:], color="tab:green")
plt.show()

