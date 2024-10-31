from time import time
import numpy as np
import random
from bsplinegenerator.bsplines import BsplineEvaluation
from bsplinegenerator.helper_functions import create_random_control_points_greater_than_angles
import matplotlib.pyplot as plt
### Control Points ###
# control_points = np.array([-3,-4,-2,-.5,1,0,2,3.5,3,6,8]) # 1 D
control_points = np.array([[-3,-4,-2,-.5,1,0,2,3.5,3,5,6.5],
                           [.5,3.5,6,5.5,3.7,2,-1,2,5,5.5,5]]) # 2 D
scale_factor = 1

### spline 2
start_time = 0
clamped = True
order = 2
num_data_points_per_interval = 10000
bspline_2 = BsplineEvaluation(control_points, order, start_time, scale_factor, clamped)
spline_data_2, time_data_2 = bspline_2.get_spline_data(num_data_points_per_interval)
spline_at_knot_points_2, knot_points_2 = bspline_2.get_spline_at_knot_points()

### spline 5
start_time = 0
clamped = False
order = 5
num_data_points_per_interval = 10000
bspline_5 = BsplineEvaluation(control_points, order, start_time, scale_factor, clamped)
spline_data_5, time_data_5 = bspline_5.get_spline_data(num_data_points_per_interval)
spline_at_knot_points_5, knot_points_5 = bspline_5.get_spline_at_knot_points()

fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].plot(spline_data_2[0,:],spline_data_2[1,:],color="tab:blue", label="B-spline")
ax[0].scatter(spline_at_knot_points_2[0,:],spline_at_knot_points_2[1,:],color="tab:blue", label="spline at knot points")
ax[0].plot(control_points[0,:],control_points[1,:],color="tab:orange")
ax[0].scatter(control_points[0,:],control_points[1,:],color="tab:orange", label="control points")
ax[0].legend()
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].set_title("Second Order B-spline")

ax[1].plot(spline_data_5[0,:],spline_data_5[1,:],color="tab:blue", label="B-spline")
ax[1].scatter(spline_at_knot_points_5[0,:],spline_at_knot_points_5[1,:],color="tab:blue", label="spline at knot points")
ax[1].plot(control_points[0,:],control_points[1,:],color="tab:orange")
ax[1].scatter(control_points[0,:],control_points[1,:],color="tab:orange", label="control points")
ax[1].legend()
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
ax[1].set_title("Fifth Order B-spline")
plt.show()

