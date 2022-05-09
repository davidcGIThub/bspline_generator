from csv import get_dialect
import numpy as np
from bsplinegenerator.helper_functions import count_number_of_control_points, find_preceding_knot_index, find_end_time, \
        get_dimension

def table_bspline_evaluation(time, control_points, knot_points, clamped = False):
    """
    This function evaluates the B spline at the given time
    """
    number_of_control_points = count_number_of_control_points(control_points)
    order = len(knot_points) - number_of_control_points - 1
    end_time = find_end_time(control_points, knot_points)
    preceding_knot_index = find_preceding_knot_index(time, order, knot_points)
    initial_control_point_index = preceding_knot_index - order
    dimension = get_dimension(control_points)
    spline_at_time_t = np.zeros((dimension,1))
    for i in range(initial_control_point_index , initial_control_point_index+order + 1):
        if dimension == 1:
            control_point = control_points[i]
        else:
            control_point = control_points[:,i][:,None]
        basis_function = cox_de_boor_table_basis_function(time, i, order, knot_points, end_time, clamped)
        spline_at_time_t += basis_function*control_point
    return spline_at_time_t

def derivative_table_bspline_evaluation(time, derivative_order, control_points, knot_points, clamped = False):
    """
    This function evaluates the B spline at the given time
    """
    number_of_control_points = count_number_of_control_points(control_points)
    order = len(knot_points) - number_of_control_points - 1
    end_time = find_end_time(control_points, knot_points)
    preceding_knot_index = find_preceding_knot_index(time, order, knot_points)
    initial_control_point_index = preceding_knot_index - order
    dimension = get_dimension(control_points)
    spline_at_time_t = np.zeros((dimension,1))
    for i in range(initial_control_point_index , initial_control_point_index+order + 1):
        if dimension == 1:
            control_point = control_points[i]
        else:
            control_point = control_points[:,i][:,None]
        basis_function_derivative = __derivative_cox_de_boor_table_basis_function(time, i, order ,derivative_order, knot_points, end_time,  clamped)
        spline_at_time_t += basis_function_derivative*control_point
    return spline_at_time_t

def cox_de_boor_table_basis_function(time, i, order , knot_points, end_time, clamped):
    table = __cox_de_boor_table_basis_function_whole_table(time, i, order, knot_points, end_time, clamped)
    return table[0,order]

def __cox_de_boor_table_basis_function_whole_table(time, i, order , knot_points, end_time, clamped):
    table = np.zeros((order+1,order+1))
    #loop through rows to create the first column
    for y in range(order+1):
        t_i_y = knot_points[i+y]
        t_i_y_1 = knot_points[i+y+1]
        if time >= t_i_y and time < t_i_y_1:
            table[y,0] = 1
        elif time == t_i_y_1 and t_i_y_1 == end_time and clamped:
            table[y,0] = 1
    # loop through remaining columns
    for kappa in range(1,order+1): 
        # loop through rows
        number_of_rows = order+1 - kappa
        for y in range(number_of_rows):
            t_i_y = knot_points[i+y]
            t_i_y_1 = knot_points[i+y+1]
            t_i_y_k = knot_points[i+y+kappa]
            t_i_y_k_1 = knot_points[i+y+kappa+1]
            horizontal_term = 0
            diagonal_term = 0
            if t_i_y_k > t_i_y:
                horizontal_term = table[y,kappa-1] * (time - t_i_y) / (t_i_y_k - t_i_y)
            if t_i_y_k_1 > t_i_y_1:
                diagonal_term = table[y+1,kappa-1] * (t_i_y_k_1 - time)/(t_i_y_k_1 - t_i_y_1)
            table[y,kappa] = horizontal_term + diagonal_term
    return table

def __derivative_cox_de_boor_table_basis_function(time, i, order, derivative_order,  knot_points, end_time,  clamped):
    if derivative_order > order:
        return 0
    table = __cox_de_boor_table_basis_function_whole_table(time, i, order , knot_points, end_time,  clamped)
    d_table = np.zeros((order+1,order+1))
    # loop through remaining columns
    for r in range(1,derivative_order+1):
        for kappa in range(1,order+1): 
            # loop through rows
            number_of_rows = order + 1 - kappa
            for y in range(number_of_rows):
                t_i_y = knot_points[i+y]
                t_i_y_1 = knot_points[i+y+1]
                t_i_y_k = knot_points[i+y+kappa]
                t_i_y_k_1 = knot_points[i+y+kappa+1]
                horizontal_term = 0
                diagonal_term = 0
                if t_i_y_k > t_i_y:
                    horizontal_term = d_table[y,kappa-1] * (time - t_i_y) / (t_i_y_k - t_i_y) + \
                        table[y,kappa-1] * (r) / (t_i_y_k - t_i_y)
                if t_i_y_k_1 > t_i_y_1:
                    diagonal_term = d_table[y+1,kappa-1] * (t_i_y_k_1 - time)/(t_i_y_k_1 - t_i_y_1) + \
                        table[y+1,kappa-1]*(-r)/(t_i_y_k_1 - t_i_y_1)
                d_table[y,kappa] = horizontal_term + diagonal_term
        if r == derivative_order:
            break
        table = d_table
        d_table = np.zeros((order+1,order+1))
    return d_table[0,order]