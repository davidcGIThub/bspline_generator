import numpy as np
from helper_functions import count_number_of_control_points

def get_bspline_arc_length(order, control_points, isClamped):
    num_control_points = count_number_of_control_points(control_points)
    num_intervals = num_control_points - order
    total_length = 0
    for i in range(num_intervals):
        interval_length = __get_bspline_interval_length(control_points, isClamped)
        total_length += interval_length
    return total_length

def __get_bspline_interval_length(control_points, isClamped):
    if isClamped:
        print("Code not implemented")
        return None
    else:
        length = __get_open_interval_length(control_points)
    return length

def __get_open_interval_length(control_points):
    order = count_number_of_control_points(control_points) - 1
    if order == 1:
        pass
    if order == 2:
        pass
    if order == 3:
        length = __get_third_order_open_interval_length(control_points)
    if order == 4:
        pass
    if order == 5:
        pass
    return length

def __get_third_order_open_interval_length(control_points):
    p0 = control_points[:,0]
    p1 = control_points[:,1]
    p2 = control_points[:,2]
    p3 = control_points[:,3]
    a = p1/2 - p0/6 - p2/2 + p3/6
    b = p0/2 - p1   + p2/2
    c = p2/2 - p0/2
    length = np.linalg.norm(a + b + c)
    return length
