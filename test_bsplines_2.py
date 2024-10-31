from time import time
import numpy as np
import random
from bsplinegenerator.bsplines import BsplineEvaluation
from bsplinegenerator.helper_functions import create_random_control_points_greater_than_angles
import matplotlib.pyplot as plt
### Control Points ###

order = 4
dimension = 2
num_control_points = order + 1
control_points = np.random.randint(10, size=(dimension,num_control_points)) # random
# control_points  = create_random_control_points_greater_than_angles(num_control_points,order,1,dimension)
scale_factor = 1

### Parameters
start_time = 0
clamped = False
num_data_points_per_interval = 10000

### Create B-Spline Object ###
bspline = BsplineEvaluation(control_points, order, start_time, scale_factor, clamped)

spline_data, time_data = bspline.get_spline_data(1000)
minvo_ctrl_pts = bspline.get_minvo_control_points()
spline_at_knots, knot_points = bspline.get_spline_at_knot_points()

# bspline.plot_spline(num_data_points_per_interval)
# bspline.plot_bezier_curves(num_data_points_per_interval)
# bspline.plot_minvo_curves(num_data_points_per_interval)
bspline.plot_minvbez_curves(num_data_points_per_interval)

