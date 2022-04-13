import numpy as np
from helper_functions import count_number_of_control_points, find_preceding_knot_index,\
    calculate_number_of_control_points, get_dimension
    
def matrix_bspline_evaluation(time, scale_factor, control_points, knot_points, clamped = False):
    """
    This function evaluates the B spline at the given time using
    the matrix method
    """
    number_of_control_points = count_number_of_control_points(control_points)
    order = len(knot_points) - number_of_control_points - 1
    preceding_knot_index = find_preceding_knot_index(time, order, knot_points)
    preceding_knot_point = knot_points[preceding_knot_index]
    initial_control_point_index = preceding_knot_index - order
    dimension = get_dimension(control_points)
    spline_at_time_t = np.zeros((dimension,1))
    i_p = initial_control_point_index
    tau = time - preceding_knot_point
    M = __get_M_matrix(i_p, order, knot_points, clamped)
    if dimension > 1:
        P = np.zeros((dimension,order+1))
    else:
        P = np.zeros(order+1)
    T = np.ones((order+1,1))
    for i in range(order+1):
        y = i
        kappa = i
        if dimension > 1:
            P[:,y] = control_points[:,i_p+y]
        else:
            P[y] = control_points[i_p+y]
        T[kappa,0] = (tau/scale_factor)**(order-kappa)
    spline_at_time_t = np.dot(P, np.dot(M,T))
    return spline_at_time_t

def derivative_matrix_bspline_evaluation(time, rth_derivative, scale_factor, control_points, knot_points, clamped = False):
    order = len(knot_points) - count_number_of_control_points(control_points) - 1
    preceding_knot_index = find_preceding_knot_index(time, order, knot_points)
    preceding_knot_point = knot_points[preceding_knot_index]
    initial_control_point_index = preceding_knot_index - order
    dimension = get_dimension(control_points)
    i_p = initial_control_point_index
    M = __get_M_matrix(i_p, order, knot_points, clamped)
    tau = (time - preceding_knot_point)
    if dimension > 1:
        P = np.zeros((dimension,order+1))
    else:
        P = np.zeros(order+1)
    for y in range(order+1):
        if dimension > 1:
            P[:,y] = control_points[:,i_p+y]
        else:
            P[y] = control_points[i_p+y]
    T = np.zeros((order+1,1))
    for i in range(order-rth_derivative+1):
        T[i,0] = (tau**(order-rth_derivative-i)*np.math.factorial(order-i)) /  (scale_factor**(order-i)*np.math.factorial(order-i-rth_derivative))
    spline_derivative_at_time_t = np.dot(P, np.dot(M,T))
    return spline_derivative_at_time_t

def __get_M_matrix(initial_control_point_index, order, knot_points, clamped):
    if order > 5:
        print("Error: Cannot compute higher than 5th order matrix evaluation")
        return None
    if order == 1:
        M = __get_1_order_matrix()
    elif clamped:
        if order == 2:
            M = __get_clamped_2_order_matrix(initial_control_point_index, order, knot_points)
        elif order == 3:
            M = __get_clamped_3_order_matrix(initial_control_point_index, order, knot_points)
        elif order == 4:
            M = __get_clamped_4_order_matrix(initial_control_point_index, order, knot_points)
        elif order == 5:
            M = __get_clamped_5_order_matrix(initial_control_point_index, order, knot_points)
    else:
        if order == 2:
            M = __get_2_order_matrix()
        elif order == 3:
            M = __get_3_order_matrix()
        elif order == 4:
            M = __get_4_order_matrix()
        elif order == 5:
            M = __get_5_order_matrix()

    return M

def __get_1_order_matrix():
    M = np.array([[-1,1],
                    [1,0]])
    return M

def __get_2_order_matrix():
    M = .5*np.array([[1,-2,1],
                        [-2,2,1],
                        [1,0,0]])
    return M

def __get_3_order_matrix():
    M = np.array([[-2 ,  6 , -6 , 2],
                    [ 6 , -12 ,  0 , 8],
                    [-6 ,  6 ,  6 , 2],
                    [ 2 ,  0 ,  0 , 0]])/12
    return M

def __get_4_order_matrix():
    M = np.array([[ 1 , -4  ,  6 , -4  , 1],
                    [-4 ,  12 , -6 , -12 , 11],
                    [ 6 , -12 , -6 ,  12 , 11],
                    [-4 ,  4  ,  6 ,  4  , 1],
                    [ 1 ,  0  ,  0 ,  0  , 0]])/24
    return M

def __get_5_order_matrix():
    M = np.array([[-1  ,  5  , -10 ,  10 , -5  , 1],
                    [ 5  , -20 ,  20 ,  20 , -50 , 26],
                    [-10 ,  30 ,  0  , -60 ,  0  , 66],
                    [ 10 , -20 , -20 ,  20 ,  50 , 26],
                    [-5  ,  5  ,  10 ,  10 ,  5  , 1 ],
                    [ 1  ,  0  ,  0  ,  0  ,  0  , 0]])/120
    return M

def __get_clamped_2_order_matrix(initial_control_point_index, order, knot_points):
    i_t = initial_control_point_index + order
    n = calculate_number_of_control_points(order, knot_points)
    M = __get_2_order_matrix()
    if i_t == 2 and knot_points[2] < knot_points[n-1]:
        M = .5*np.array([[2,-4,2],
                        [-3,4,0],
                        [1,0,0]])
    elif i_t == n - 1 and knot_points[2] < knot_points[n-1]:
        M = .5*np.array([[1,-2,1],
                        [-3,2,1],
                        [2,0,0]])
    elif i_t == 2 and knot_points[2] == knot_points[n-1] :
        M = 0.5*np.array([[2,-4,2],
                            [-4,4,0],
                            [2,0,0]])
    return M

def __get_clamped_3_order_matrix(initial_control_point_index, order, knot_points):
    i_t = initial_control_point_index + order
    n = calculate_number_of_control_points(order, knot_points)
    M = __get_3_order_matrix()
    if i_t == 3 and knot_points[3] < knot_points[n-1]:
        M = np.array([[-12 ,  36 , -36 , 12],
                    [ 21 , -54 ,  36 , 0],
                    [-11 ,  18 ,  0  , 0],
                    [ 2  ,  0  ,  0  , 0]])/12.0
    elif i_t == 4 and knot_points[4] < knot_points[n-2]:
        M = np.array([[-3 ,  9 , -9 , 3],
                    [ 7 , -15 ,  3 , 7],
                    [-6 ,  6 , 6 , 2],
                    [ 2 ,  0 ,  0 , 0]])/12

    elif i_t == n - 2 and knot_points[4] < knot_points[n-2]:
        M = np.array([[-2  , 6   ,  -6 , 2],
                    [6 , -12  , 0 ,  8],
                    [ -7,    6 ,  6 , 2 ],
                    [ 3,  0   , 0  , 0]])/12
    elif i_t == n - 1 and knot_points[3] < knot_points[n-1]:
        M = np.array([[-2  , 6   ,  -6 , 2],
                    [11 , -15  , -3 ,  7],
                    [-21,  9   ,  9 ,  3],
                    [12 ,  0   ,  0 ,  0]])/12.0
    elif i_t == 4 and knot_points[4] == knot_points[n-2]:
        M = np.array([[-3 , 9 , -9 , 3],
            [7, -15 , 3 , 7],
            [-7 , 6 , 6 , 2],
            [3 , 0 , 0 , 0]])/12
    elif i_t == 3 and knot_points[3] == knot_points[n-1]:
            M = np.array([[-12 , 36 , -36 , 12],
                [36 , -72 , 36 , 0],
                [-36 , 36 , 0 , 0],
                [12 , 0 , 0 , 0]])/12
    return M

def __get_clamped_4_order_matrix(initial_control_point_index):
    M = __get_4_order_matrix()
    # TODOD
    return M

def __get_clamped_5_order_matrix(initial_control_point_index):
    M = __get_5_order_matrix()
    return M