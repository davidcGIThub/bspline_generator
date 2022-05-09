import numpy as np
from bsplinegenerator.helper_functions import count_number_of_control_points, find_preceding_knot_index, find_end_time, get_dimension


def recursive_bspline_evaluation(time, control_points, knot_points, clamped = False):
    """
    This function evaluates the B spline at the given time
    """
    number_of_control_points = count_number_of_control_points(control_points)
    order = len(knot_points) - number_of_control_points - 1
    preceding_knot_index = find_preceding_knot_index(time, order, knot_points)
    initial_control_point_index = preceding_knot_index - order
    end_time = find_end_time(control_points, knot_points)
    dimension = get_dimension(control_points)
    spline_at_time_t = np.zeros((dimension,1))
    for i in range(initial_control_point_index , initial_control_point_index+order + 1):
        if dimension == 1:
            control_point = control_points[i]
        else:
            control_point = control_points[:,i][:,None]
        basis_function = __cox_de_boor_recursion_basis_function(time, i, order, knot_points, end_time,  clamped)
        spline_at_time_t += basis_function*control_point
    return spline_at_time_t
        
def derivative_recursive_bspline_evaluation(time, r, control_points, knot_points, clamped = False):
    """
    This function evaluates the B spline at the given time
    """
    number_of_control_points = count_number_of_control_points(control_points)
    order = len(knot_points) - number_of_control_points - 1
    preceding_knot_index = find_preceding_knot_index(time, order, knot_points)
    initial_control_point_index = preceding_knot_index - order
    end_time = find_end_time(control_points, knot_points)
    dimension = get_dimension(control_points)
    derivative_at_time_t = np.zeros((dimension,1))
    for i in range(initial_control_point_index , initial_control_point_index+order + 1):
        if dimension == 1:
            control_point = control_points[i]
        else:
            control_point = control_points[:,i][:,None]
        basis_function_derivative = __derivative_cox_de_boor_recursion_basis_function(time, i,order, r, knot_points, end_time,  clamped)
        derivative_at_time_t += basis_function_derivative*control_point
    return derivative_at_time_t

def __cox_de_boor_recursion_basis_function(time, i, kappa ,knot_points, end_time,  clamped):
    t = time
    t_i = knot_points[i]
    t_i_1 = knot_points[i+1]
    t_i_k = knot_points[i+kappa]
    t_i_k_1 = knot_points[i+kappa+1]
    if kappa == 0:
        if (t >= t_i and t < t_i_1):
            basis_function = 1
        elif (clamped and t == end_time and t >= t_i and t == t_i_1):
            basis_function = 1
        else:
            basis_function = 0
    else:
        if t_i < t_i_k:
            term1 = (t - t_i) / (t_i_k - t_i) * __cox_de_boor_recursion_basis_function(t,i,kappa-1, knot_points, end_time, clamped)
        else:
            term1 = 0
        if t_i_1 < t_i_k_1:
            term2 = (t_i_k_1 - t)/(t_i_k_1 - t_i_1) * __cox_de_boor_recursion_basis_function(t,i+1,kappa-1, knot_points, end_time, clamped)
        else:
            term2 = 0
        basis_function = term1 + term2
    return basis_function

def __derivative_cox_de_boor_recursion_basis_function(time, i, kappa, r, knot_points, end_time,  clamped):
    if r == 0:
        return __cox_de_boor_recursion_basis_function(time, i, kappa, knot_points, end_time,  clamped)
    t = time
    t_i = knot_points[i]
    t_i_1 = knot_points[i+1]
    t_i_k = knot_points[i+kappa]
    t_i_k_1 = knot_points[i+kappa+1]
    if kappa == 0:
        basis_function = 0
    else:
        if t_i < t_i_k:
            if kappa-1 > 0:
                term1 = (t - t_i) / (t_i_k - t_i) * __derivative_cox_de_boor_recursion_basis_function(t,i,kappa-1,r, knot_points, end_time,  clamped) + \
                    (r) / (t_i_k - t_i) * __derivative_cox_de_boor_recursion_basis_function(t,i,kappa-1, r-1, knot_points, end_time,  clamped)
            else:
                term1 = (1) / (t_i_k - t_i) * __derivative_cox_de_boor_recursion_basis_function(t,i,kappa-1, r-1, knot_points, end_time,  clamped)
        else:
            term1 = 0
        if t_i_1 < t_i_k_1:
            if kappa-1 > 0:
                term2 = (t_i_k_1 - t)/(t_i_k_1 - t_i_1) * __derivative_cox_de_boor_recursion_basis_function(t,i+1,kappa-1,r, knot_points, end_time,  clamped) + \
                    (-r)/(t_i_k_1 - t_i_1) * __derivative_cox_de_boor_recursion_basis_function(t,i+1,kappa-1,r-1, knot_points, end_time,  clamped)
            else:
                term2 = (-1)/(t_i_k_1 - t_i_1) * __derivative_cox_de_boor_recursion_basis_function(t,i+1,kappa-1,r-1, knot_points, end_time,  clamped)
        else:
            term2 = 0
        basis_function = term1 + term2
    return basis_function