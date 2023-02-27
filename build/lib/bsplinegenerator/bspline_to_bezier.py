"""
This module contains code that converts b-spline control points
to Bezier curve control points
"""
import numpy as np
from bsplinegenerator.helper_functions import count_number_of_control_points

def convert_to_bezier_control_points(bspline_control_points):
    number_of_control_points = count_number_of_control_points(bspline_control_points)
    order = number_of_control_points - 1
    if order > 5:
        raise Exception("Can only retrieve conversion matrix for curves of order 1-5")
    conversion_matrix = get_bspline_to_bezier_conversion_matrix(order)
    bezier_control_points = np.transpose(np.dot(conversion_matrix, np.transpose(bspline_control_points)))
    return bezier_control_points

def convert_list_to_bezier_control_points(bspline_control_points,order):
    if order > 5:
        raise Exception("Can only retrieve conversion matrix for curves of order 1-5")
    number_of_bspline_control_points = count_number_of_control_points(bspline_control_points)
    number_of_knot_point_segments = number_of_bspline_control_points - order
    number_of_bezier_control_points = (number_of_knot_point_segments)*order+1
    composite_conversion_matrix = np.zeros((number_of_bezier_control_points,number_of_bspline_control_points))
    conversion_matrix = get_bspline_to_bezier_conversion_matrix(order)
    for i in range(number_of_knot_point_segments):
        composite_conversion_matrix[i*order:i*order+order+1 , i:i+order+1] = conversion_matrix
    bezier_control_point_list = np.transpose(np.dot(composite_conversion_matrix, np.transpose(bspline_control_points)))
    return bezier_control_point_list

def get_bspline_to_bezier_conversion_matrix(order):
    conversion_matrix = np.array([])

    if order == 1:
        conversion_matrix = np.array([[1,0],
                                      [0,1]])
    elif order == 2:
        conversion_matrix = np.array([[1,1,0],
                                      [0,2,0],
                                      [0,1,1]])/2                         
    elif order == 3:
        conversion_matrix = np.array([[1,4,1,0],
                                      [0,4,2,0],
                                      [0,2,4,0],
                                      [0,1,4,1]])/6
    elif order == 4:
        conversion_matrix = np.array([[1, 11, 11, 1, 0],
                                      [0, 8, 14, 2, 0],
                                      [0, 4, 16, 4, 0],
                                      [0, 2, 14, 8, 0],
                                      [0, 1, 11, 11, 1]])/24
    elif order == 5:
        conversion_matrix = np.array([[1,26,66,26,1,0],
                                      [0,16,66,36,2,0],
                                      [0,8,60,48,4,0],
                                      [0,4,48,60,8,0],
                                      [0,2,36,66,16,0],
                                      [0,1,26,66,26,1]])/120
    else:
        raise Exception("Can only retrieve conversion matrix for curves of order 1-5")
    return conversion_matrix