"""
This module contains code that converts b-spline control points
to Bezier curve control points
"""
import numpy as np
from bsplinegenerator.helper_functions import count_number_of_control_points

def convert_to_bezier_control_points(bspline_control_points):
    number_of_control_points = count_number_of_control_points(bspline_control_points)
    order = number_of_control_points - 1
    if order > 7:
        raise Exception("Can only retrieve conversion matrix for curves of order 1-7")
    conversion_matrix = get_bspline_to_bezier_conversion_matrix(order)
    bezier_control_points = np.transpose(np.dot(conversion_matrix, np.transpose(bspline_control_points)))
    return bezier_control_points

def convert_list_to_bezier_control_points(bspline_control_points,order):
    number_of_bspline_control_points = count_number_of_control_points(bspline_control_points)
    composite_conversion_matrix = get_composite_bspline_to_bezier_conversion_matrix(number_of_bspline_control_points,order)
    bezier_control_point_list = np.transpose(np.dot(composite_conversion_matrix, np.transpose(bspline_control_points)))
    return bezier_control_point_list

def get_composite_bspline_to_bezier_conversion_matrix(num_bspline_control_points, order):
    if order > 7:
        raise Exception("Can only retrieve conversion matrix for curves of order 1-7")
    number_of_knot_point_segments = num_bspline_control_points - order
    number_of_bezier_control_points = (number_of_knot_point_segments)*order+1
    composite_conversion_matrix = np.zeros((number_of_bezier_control_points,num_bspline_control_points))
    conversion_matrix = get_bspline_to_bezier_conversion_matrix(order)
    for i in range(number_of_knot_point_segments):
        composite_conversion_matrix[i*order:i*order+order+1 , i:i+order+1] = conversion_matrix
    return composite_conversion_matrix

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
    elif order == 6:
        conversion_matrix = 1/720*np.array([[1  , 0  , 0  , 0  , 0  , 0  , 0],
                                      [57 , 32 , 16 , 8  , 4  , 2  , 1],
                                      [302, 262, 212, 160, 116, 82 , 57],
                                      [302, 342, 372, 384, 372, 342, 302],
                                      [57 , 82 , 116, 160, 212, 262, 302],
                                      [1  , 2  , 4  , 8  , 16 , 32 , 57],
                                      [0  , 0  , 0  , 0  , 0  , 0  , 1]]).T
    elif order == 7:
        conversion_matrix = 1/5040*np.array([[ 1, 0, 0,  0,  0, 0, 0, 0],
                                    [ 120,   64,   32,   16,   8, 4, 2, 1],
                                    [ 1191,    946,     716,      520,    368,   256,   176,   120],
                                    [ 2416,    2416,     2336,     2176,     1952,    1696,    1436,    1191],
                                    [ 1191,    1436,     1696,     1952,     2176,    2336,    2416,    2416],
                                    [ 120,   176,    256,    368,     520,    716,    946,    1191],
                                    [ 1,  2,  4,   8,   16,  32,   64,   120],
                                    [ 0, 0, 0, 0, 0, 0, 0,  1]]).T
    else:
        raise Exception("Can only retrieve conversion matrix for curves of order 1-7")
    return conversion_matrix