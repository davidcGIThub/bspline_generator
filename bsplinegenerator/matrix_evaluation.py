import numpy as np
from bsplinegenerator.helper_functions import count_number_of_control_points, find_preceding_knot_index,\
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
    M = get_M_matrix(i_p, order, knot_points, clamped)
    if dimension > 1:
        P = np.zeros((dimension,order+1))
    else:
        P = np.zeros(order+1)
    for i in range(order+1):
        if dimension > 1:
            P[:,i] = control_points[:,i_p+i]
        else:
            P[i] = control_points[i_p+i]
    T = get_T_vector(order,time,preceding_knot_point,scale_factor)
    spline_at_time_t = np.dot(P, np.dot(M,T))
    return spline_at_time_t

def derivative_matrix_bspline_evaluation(time, rth_derivative, scale_factor, control_points, knot_points, clamped = False):
    order = len(knot_points) - count_number_of_control_points(control_points) - 1
    preceding_knot_index = find_preceding_knot_index(time, order, knot_points)
    preceding_knot_point = knot_points[preceding_knot_index]
    initial_control_point_index = preceding_knot_index - order
    dimension = get_dimension(control_points)
    i_p = initial_control_point_index
    M = get_M_matrix(i_p, order, knot_points, clamped)
    if dimension > 1:
        P = np.zeros((dimension,order+1))
    else:
        P = np.zeros(order+1)
    for y in range(order+1):
        if dimension > 1:
            P[:,y] = control_points[:,i_p+y]
        else:
            P[y] = control_points[i_p+y]
    T = get_T_derivative_vector(order,time,preceding_knot_point,rth_derivative,scale_factor)
    spline_derivative_at_time_t = np.dot(P, np.dot(M,T))
    return spline_derivative_at_time_t

def matrix_bspline_evaluation_for_dataset(control_points, knot_points, num_points_per_interval, clamped = False):
    """
    This function evaluates the B spline for a given time data-set
    """
    #initialize variables
    num_ppi = num_points_per_interval
    dimension = get_dimension(control_points)
    number_of_control_points = count_number_of_control_points(control_points)
    order = len(knot_points) - number_of_control_points - 1
    num_intervals = number_of_control_points - order
    #create steps matrix
    steps_array = np.linspace(0,1,num_ppi+1)
    L = np.ones((order+1,num_ppi+1))
    for i in range(order+1):
        L[i,:] = steps_array**(order-i)
    # Find M matrix
    M = get_M_matrix(0, order, [], False)
    #Evaluate spline data
    if dimension > 1:
        spline_data = np.zeros((dimension,num_intervals*num_ppi+1))
    else:
        spline_data = np.zeros(num_intervals*num_ppi+1)
    for i in range(num_intervals):
        if dimension > 1:
            P = control_points[:,i:i+order+1]
        else:
            P = control_points[i:i+order+1]
        #Find M matrix if clamped
        if clamped:
            M = get_M_matrix(i, order, knot_points, True)
        spline_data_over_interval = np.dot(np.dot(P,M),L)
        if dimension > 1:
            if i == num_intervals-1:
                spline_data[:,i*num_ppi:(i+1)*num_ppi+1] = spline_data_over_interval[:,0:num_ppi+1]
            else:
                spline_data[:,i*num_ppi:(i+1)*num_ppi] = spline_data_over_interval[:,0:num_ppi]
        else:
            if i == num_intervals-1:
                spline_data[i*num_ppi:(i+1)*num_ppi+1] = spline_data_over_interval[0:num_ppi+1]
            else:
                spline_data[i*num_ppi:(i+1)*num_ppi] = spline_data_over_interval[0:num_ppi]
    return spline_data

def matrix_bspline_derivative_evaluation_for_dataset(derivative_order, scale_factor, control_points, knot_points, num_points_per_interval, clamped = False):
    """
    This function evaluates the B spline for a given time data-set
    """
    # Initialize variables
    num_ppi = num_points_per_interval
    dimension = get_dimension(control_points)
    number_of_control_points = count_number_of_control_points(control_points)
    order = len(knot_points) - number_of_control_points - 1
    num_intervals = number_of_control_points - order
    #create steps matrix
    steps_array = np.linspace(0,1,num_ppi+1)
    L_r = np.zeros((order+1,num_ppi+1))
    # Find M matrix
    M = get_M_matrix(0, order, [], False)
    K = __create_k_matrix(order,derivative_order,scale_factor)
    for i in range(order-derivative_order+1):
        L_r[i,:] = steps_array**(order-derivative_order-i)
    # Evaluate Spline data
    if dimension > 1:
        spline_derivative_data = np.zeros((dimension,num_intervals*num_ppi+1))
    else:
        spline_derivative_data = np.zeros(num_intervals*num_ppi+1)
    for i in range(num_intervals):
        if dimension > 1:
            P = control_points[:,i:i+order+1]
        else:
            P = control_points[i:i+order+1]
        # Find M matrix if clamped
        if clamped:
            M = get_M_matrix(i, order, knot_points, True)
        spline_derivative_data_over_interval = np.dot(np.dot(P,M),np.dot(K,L_r))
        if dimension > 1:
            if i == num_intervals-1:
                spline_derivative_data[:,i*num_ppi:(i+1)*num_ppi+1] = spline_derivative_data_over_interval[:,0:num_ppi+1]
            else:
                spline_derivative_data[:,i*num_ppi:(i+1)*num_ppi] = spline_derivative_data_over_interval[:,0:num_ppi]
        else:
            if i == num_intervals-1:
                spline_derivative_data[i*num_ppi:(i+1)*num_ppi+1] = spline_derivative_data_over_interval[0:num_ppi+1]
            else:
                spline_derivative_data[i*num_ppi:(i+1)*num_ppi] = spline_derivative_data_over_interval[0:num_ppi]
    return spline_derivative_data

def __create_k_matrix(order,derivative_order,scale_factor):
    K = np.zeros((order+1,order+1))
    for i in range(order-derivative_order+1):
        K[i,i] = np.math.factorial(order-i)/np.math.factorial(order-derivative_order-i)
    K = K/scale_factor**(derivative_order)
    return K

def get_M_matrix(initial_control_point_index, order, knot_points, clamped):
    if order == 0:
        return 1
    if order == 1:
        M = __get_1_order_matrix()
    elif clamped:
        if order > 5:
            print("Error: Cannot compute higher than 5th order matrix evaluation for clamped spline")
            return None
        elif order == 2:
            M = __get_clamped_2_order_matrix(initial_control_point_index, order, knot_points)
        elif order == 3:
            M = __get_clamped_3_order_matrix(initial_control_point_index, order, knot_points)
        elif order == 4:
            M = __get_clamped_4_order_matrix(initial_control_point_index, order, knot_points)
        elif order == 5:
            M = __get_clamped_5_order_matrix(initial_control_point_index, order, knot_points)
    else:
        if order > 7:
            print("Error: Cannot compute higher than 7th order matrix evaluation for open spline")
            return None
        elif order == 2:
            M = __get_2_order_matrix()
        elif order == 3:
            M = __get_3_order_matrix()
        elif order == 4:
            M = __get_4_order_matrix()
        elif order == 5:
            M = __get_5_order_matrix()
        elif order == 6:
            M = __get_6_order_matrix()
        elif order == 7:
            M = __get_7_order_matrix()
    return M

def get_T_derivative_vector(order,t,tj,rth_derivative,scale_factor):
    T = np.zeros((order+1,1))
    t_tj = t-tj
    for i in range(order-rth_derivative+1):
        T[i,0] = (t_tj**(order-rth_derivative-i))/(scale_factor**(order-i)) * np.math.factorial(order-i)/np.math.factorial(order-i-rth_derivative)
    return T

def get_T_vector(order,t,tj,scale_factor):
    T = np.ones((order+1,1))
    t_tj = t-tj
    for i in range(order+1):
        if i > order:
            T[i,0] = 0
        else:
            T[i,0] = (t_tj/scale_factor)**(order-i)
    return T

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
    M = np.array([[-2 ,  6  , -6 , 2],
                  [ 6 , -12 ,  0 , 8],
                  [-6 ,  6  ,  6 , 2],
                  [ 2 ,  0  ,  0 , 0]])/12
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

def __get_6_order_matrix():
    M = np.array([[1, -6 , 15 , -20 , 15 , -6 , 1],
                    [-6, 30, -45, -20, 135, -150, 57],
                    [15, -60, 30, 160, -150, -240, 302],
                    [-20, 60, 30, -160, -150, 240, 302],
                    [15, -30, -45, 20, 135, 150, 57],
                    [-6, 6, 15, 20, 15, 6, 1],
                    [1, 0, 0 , 0, 0, 0, 0]])/720
    return M
    
def __get_7_order_matrix():
    M = np.array(  [[-1, 7, -21, 35, -35, 21, -7, 1],
                    [7, -42, 84, 0, -280, 504, -392, 120],
                    [-21, 105, -105, -315, 665, 315, -1715, 1191],
                    [35, -140, 0, 560, 0, -1680, 0, 2416],
                    [-35, 105, 105, -315, -665, 315, 1715, 1191],
                    [21, -42, -84, 0,  280, 504, 392, 120],
                    [-7, 7, 21, 35, 35, 21, 7 , 1],
                    [1, 0, 0, 0 , 0, 0, 0, 0]])/5040
    return M

def __get_clamped_2_order_matrix(initial_control_point_index, order, knot_points):
    i_t = initial_control_point_index + order
    n = calculate_number_of_control_points(order, knot_points)
    M = np.array([])
    if n == 3:
        M = 0.5*np.array([[2,-4,2],
                            [-4,4,0],
                            [2,0,0]])
    elif n >= 4:
        if i_t == 2:
            M = .5*np.array([[2,-4,2],
                            [-3,4,0],
                            [1,0,0]])
        elif i_t == n - 1:
            M = .5*np.array([[1,-2,1],
                            [-3,2,1],
                            [2,0,0]])
        else:
            M = __get_2_order_matrix()
    return M

def __get_clamped_3_order_matrix(initial_control_point_index, order, knot_points):
    i_t = initial_control_point_index + order
    n = calculate_number_of_control_points(order, knot_points)
    M = np.array([])
    if n == 4:
        M = np.array([[-12 , 36 , -36 , 12],
            [36 , -72 , 36 , 0],
            [-36 , 36 , 0 , 0],
            [12 , 0 , 0 , 0]])/12
    elif n == 5:
        if i_t == 3:
            M = np.array([[-12 ,  36 , -36 , 12],
                          [ 21 , -54 ,  36 , 0],
                          [-12 ,  18 ,  0  , 0],
                          [ 3  ,  0  ,  0  , 0]])/12.0
        elif i_t == 4:
            M = np.array([[-3  , 9   ,  -9 , 3],
                          [12,  -18   ,  0 ,  6],
                          [-21,  9   ,  9 ,  3],
                          [12 ,  0   ,  0 ,  0]])/12.0
    elif n >= 6:
        if i_t == 3:
            M = np.array([[-12 ,  36 , -36 , 12],
                [ 21 , -54 ,  36 , 0],
                [-11 ,  18 ,  0  , 0],
                [ 2  ,  0  ,  0  , 0]])/12.0
        elif i_t == 4:
            if knot_points[4] == knot_points[n-2]:
                M = np.array([[-3 , 9 , -9 , 3],
                        [7, -15 , 3 , 7],
                        [-7 , 6 , 6 , 2],
                        [3 , 0 , 0 , 0]])/12.0
            elif knot_points[4] < knot_points[n-2]:
                M = np.array([[-3 ,  9 , -9 , 3],
                        [ 7 , -15 ,  3 , 7],
                        [-6 ,  6 , 6 , 2],
                        [ 2 ,  0 ,  0 , 0]])/12
        elif i_t == n-2 and knot_points[4] < knot_points[n-2]:
            M = np.array([[-2  , 6   ,  -6 , 2],
                    [6 , -12  , 0 ,  8],
                    [ -7,    6 ,  6 , 2 ],
                    [ 3,  0   , 0  , 0]])/12
        elif i_t == n-1:
            M = np.array([[-2  , 6   ,  -6 , 2],
                    [11 , -15  , -3 ,  7],
                    [-21,  9   ,  9 ,  3],
                    [12 ,  0   ,  0 ,  0]])/12.0
        else:
            M = __get_3_order_matrix()
    return M

def __get_clamped_4_order_matrix(initial_control_point_index, order, knot_points):
    i_t = initial_control_point_index + order
    n = calculate_number_of_control_points(order, knot_points)
    M = np.array([])
    if n == 5:
        M = np.array([[1  ,  -4 , 6   ,-4 , 1],
                     [ -4 ,  12 , -12 , 4 , 0],
                     [  6 , -12 ,  6  , 0 , 0],
                     [ -4 ,  4  , 0   , 0 , 0],
                     [1   ,   0 ,   0 , 0 , 0]])
    elif n==6:
        if i_t == 4:
            M = np.array([[8  , -32, 48 , -32, 8],
                        [-15, 56 , -72, 32 , 0],
                        [11 , -32, 24 , 0 , 0],
                        [-5 , 8 , 0 , 0 , 0],
                        [1 , 0 , 0 , 0 , 0]])/8.0
        elif i_t == 5:
            M = np.array([[1  , -4, 6 , -4, 1],
                        [-5, 12 , -6, -4 , 3],
                        [11 , -12, -6 , 4 , 3],
                        [-15 , 4 , 6 , 4 , 1],
                        [8 , 0 , 0 , 0 , 0]])/8.0
    elif n==7:
        if i_t == 4:
            M = np.array([[72, -288, 432, -288, 72],
                        [-135, 504, -648, 288, 0],
                        [85, -264, 216, 0 , 0],
                        [-26, 48, 0, 0 ,0],
                        [4, 0 , 0 , 0, 0]])/72.0
        elif i_t == 5:
            M = np.array([[9, -36, 54, -36, 9],
                        [-23, 76, -66, -20, 37],
                        [28, -56, -12, 40 , 22],
                        [-23, 16, 24, 16 ,4],
                        [9, 0 , 0 , 0, 0]])/72.0
        elif i_t == 6:
            M = np.array([[4 , -16 , 24 , -16 , 4],
                          [-26 , 56 , -12 , -40 , 22],
                          [85 , -76 , -66 , 20 , 37],
                          [-135 , 36 , 54 , 36 , 9],
                          [72 , 0 , 0 , 0 , 0]])/72
    elif n>7:
        if i_t == 4:
            M = np.array([[72, -288, 432, -288, 72],
                        [-135, 504, -648, 288, 0],
                        [85, -264, 216, 0 , 0],
                        [-25, 48, 0, 0 ,0],
                        [3, 0 , 0 , 0, 0]])/72.0
        elif i_t == 5:
            if knot_points[5] == knot_points[n-3]:
                M = np.array([[9 , -36 , 54 , -36, 9],
                            [-23, 76, -66, -20, 37],
                            [23, -52, -6, 44, 23],
                            [-13, 12, 18, 12, 3],
                            [4, 0 , 0 , 0, 0]])/72.0
            elif knot_points[5] < knot_points[n-3]:
                M = np.array([[9 , -36 , 54 , -36, 9],
                            [-23, 76, -66, -20, 37],
                            [23, -52, -6, 44, 23],
                            [-12, 12, 18, 12, 3],
                            [3, 0 , 0 , 0, 0]])/72.0
        elif i_t == 6:
            if knot_points[6] == knot_points[n-2]:
                M = np.array([[4, -16, 24, -16, 4],
                            [-13, 40, -24, -32, 32],
                            [23, -40, -24, 32, 32],
                            [-23, 16, 24, 16, 4],
                            [9, 0 , 0, 0 , 0]])/72.0
            elif knot_points[6] == knot_points[n-3]:
                M = np.array([[4,-16,24,-16,4],
                              [-13,40,-24,-32,32],
                              [18,-36,-18,36,33],
                              [-13,12,18,12,3],
                              [4,0,0,0,0]])/72.0
            elif knot_points[6] < knot_points[n-3]:
                M = np.array([[4,-16,24,-16,4],
                              [-13,40,-24,-32,32],
                              [18,-36,-18,36,33],
                              [-12,12,18,12,3],
                              [3,0,0,0,0]])/72.0
        elif i_t == n-3 and knot_points[6] < knot_points[n-3]:
                M = np.array([[3, -12, 18, -12, 3],
                              [-12, 36, -18, -36, 33],
                              [18, -36, -18, 36, 33],
                              [-13, 12, 18, 12, 3],
                              [4, 0 , 0 , 0 , 0]])/72.0
        elif i_t == n-2 and knot_points[6] < knot_points[n-2]:
                M = np.array([[3, -12, 18, -12, 3],
                            [-12, 36, -18, -36, 33],
                            [23, -40, -24, 32, 32],
                            [-23, 16, 24, 16, 4],
                            [9, 0 , 0, 0 , 0]])/72.0
        elif i_t == n-1:
            M = np.array([[3, -12, 18, -12, 3],
                          [-25, 52, -6, -44, 23],
                          [85, -76, -66, 20, 37],
                          [-135, 36, 54, 36, 9],
                          [72, 0 , 0 , 0 ,0]])/72.0
        else:
            M = __get_4_order_matrix()
    return M


def __get_clamped_5_order_matrix(initial_control_point_index, order, knot_points):
    i_t = initial_control_point_index + order
    n = calculate_number_of_control_points(order, knot_points)
    M = np.array([])
    if n == 6:
        M = np.array([[-1, 5, -10, 10, -5, 1],
                      [5, -20 , 30 , -20, 5, 0],
                      [-10, 30 , -30, 10, 0 , 0],
                      [10, -20, 10, 0 , 0 , 0],
                      [-5, 5 , 0 , 0 , 0, 0],
                      [1, 0 , 0 , 0 , 0 , 0]])
    if n == 7:
        if i_t == 5:
            M = np.array([[-16, 80, -160, 160, -80, 16],
                      [31, -150 , 280 , -240, 80, 0],
                      [-26, 110 , -160, 80, 0 , 0],
                      [16, -50, 40, 0 , 0 , 0],
                      [-6, 10 , 0 , 0 , 0, 0],
                      [1, 0 , 0 , 0 , 0 , 0]])/16
        elif i_t == 6:
            M = np.array([[-1, 5, -10, 10, -5, 1],
                      [6, -20 , 20 , 0, -10, 4],
                      [-16, 30 , 0, -20, 0 , 6],
                      [26, -20, -20, 0 , 10 , 4],
                      [-31, 5 , 10 , 10 , 5, 1],
                      [16, 0 , 0 , 0 , 0 , 0]])/16
    elif n == 8:
        if i_t == 5:
            M = np.array([[-432, 2160, -4320, 4320, -2160, 432],
                          [837, -4050, 7560, -6480, 2160, 0],
                          [-575, 2550, -3960, 2160, 0 , 0],
                          [222, -780, 720, 0 , 0, 0],
                          [-60, 120, 0 , 0 , 0 , 0],
                          [8, 0 , 0 , 0 , 0 , 0]])/432.0
        elif i_t == 6:
            M = np.array([[-27, 135, -270, 270, -135, 27],
                          [73, -325, 490, -170, -235, 175],
                          [-102, 330, -180, -300, 150 , 162],
                          [102, -180, -120, 120 , 180, 60],
                          [-73, 40, 80 , 80 , 40 , 8],
                          [27, 0 , 0 , 0 , 0 , 0]])/432.0
        elif i_t == 7:
            M = np.array([[-8, 40, -80, 80, -40, 8],
                          [60, -180, 120, 120, -180, 60],
                          [-222, 330, 180, -300, -150, 162],
                          [575, -325, -490, -170, 235, 175],
                          [-837, 135, 270, 270, 135, 27],
                          [432, 0 , 0 , 0 , 0, 0]])/432.0
    elif n == 9:
        if i_t == 5:
            M = np.array([[-864, 4320, -8640, 8640, -4320, 864],
                          [1674, -8100, 15120, -12960, 4320, 0],
                          [-1150, 5100, -7920, 4320, 0, 0],
                          [415, -1500, 1440, 0 , 0 , 0],
                          [-84, 180, 0 , 0 , 0 , 0],
                          [9 , 0 , 0 , 0 , 0 , 0]])/864
        elif i_t == 6:
            M = np.array([[-54, 270, -540, 540, -270, 54],
                          [146, -650, 980, -340, -470, 350],
                          [-161, 575, -410, -530, 395, 355],
                          [108, -240, -120, 240, 300, 96],
                          [-55, 45, 90, 90, 45, 9],
                          [16, 0 , 0 , 0 , 0, 0]])/864
        elif i_t == 7:
            M = np.array([[-16, 80, -160, 160, -80, 16],
                          [55, -230, 280, 80, -400, 224],
                          [-108, 300, 0 , -480, 0, 384],
                          [161, -230, -280, 80, 400, 224],
                          [-146, 80, 160, 160, 80, 16],
                          [54, 0 , 0 , 0 , 0 , 0]])/864.0
        elif i_t == 8:
            M = np.array([[-9, 45, -90, 90, -45, 9],
                          [84, -240, 120, 240, -300, 96],
                          [-415, 575, 410, -530, -395, 355],
                          [1150, -650, -980, -340, 470, 350],
                          [-1674, 270, 540, 540, 270, 54],
                          [864, 0 , 0 , 0 , 0 , 0]])/864.0
    elif n > 9:
        if i_t == 5:
            M = np.array([[-4320, 21600, -43200, 43200, -21600, 4320],
                          [8370, -40500, 75600, -64800, 21600, 0],
                          [-5750, 25500, -39600, 21600 , 0, 0],
                          [2075, -7500, 7200, 0 , 0 , 0],
                          [-411, 900, 0 , 0 , 0 , 0],
                          [36, 0 , 0 , 0 , 0 , 0]])/4320
        elif i_t == 6:
            if knot_points[6] == knot_points[n-4]:
                M = np.array([[-270, 1350, -2700, 2700, -1350, 270],
                              [730, -3250, 4900, -1700, -2350, 1750],
                              [-805, 2875, -2050, -2650, 1975, 1775],
                              [489, -1155, -510, 1290, 1545, 489],
                              [-189, 180, 360, 360, 180, 36],
                              [45, 0 , 0, 0, 0, 0]])/4320
            elif knot_points[6] < knot_points[n-4]:
                M = np.array([[-270, 1350, -2700, 2700, -1350, 270],
                              [730, -3250, 4900, -1700, -2350, 1750],
                              [-805, 2875, -2050, -2650, 1975, 1775],
                              [489, -1155, -510, 1290, 1545, 489],
                              [-180, 180, 360, 360, 180, 36],
                              [36, 0 , 0, 0, 0 , 0]])/4320
        elif i_t == 7:
            if knot_points[7] == knot_points[n-3]:
                M = np.array([[-80, 400, -800, 800, -400, 80],
                              [275, -1150, 1400, 400, -2000, 1120],
                              [-411, 1290, -240, -2280, 420, 2148],
                              [411, -765, -810, 630, 1755, 927],
                              [-275, 225, 450, 450, 225, 45],
                              [80, 0 , 0 , 0 , 0 , 0]])/4320
            elif knot_points[7] == knot_points[n-4]:
                M = np.array([[-80, 400, -800, 800, -400, 80],
                              [275, -1150, 1400, 400, -2000, 1120],
                              [-411, 1290, -240, -2280, 420, 2148],
                              [360, -720, -720, 720, 1800, 936],
                              [-189, 180, 360, 360, 180, 36],
                              [45, 0 , 0 , 0 , 0 , 0]])/4320
            elif knot_points[7] < knot_points[n-4]:
                M = np.array([[-80, 400, -800, 800, -400, 80],
                              [275, -1150, 1400, 400, -2000, 1120],
                              [-411, 1290, -240, -2280, 420, 2148],
                              [360, -720, -720, 720, 1800, 936],
                              [-180, 180, 360, 360, 180, 36],
                              [36, 0, 0, 0, 0, 0]])/4320
        elif i_t == 8:
            if knot_points[8] == knot_points[n-2]:
                M = np.array([[-45, 225, -450, 450, -225, 45],
                              [189, -765, 810, 630, -1755, 927],
                              [-489, 1290, 240, -2280, -420, 2148],
                              [805, -1150, -1400, 400, 2000, 1120],
                              [-730, 400, 800, 800, 400, 80],
                              [270, 0 , 0 , 0 , 0 , 0]])/4320
            elif knot_points[8] == knot_points[n-3]:
                M = np.array([[-45, 225, -450, 450, -225, 45],
                            [189, -765, 810, 630, -1755, 927],
                            [-360, 1080, 0, -2160, 0, 2376],
                            [411, -765, -810, 630, 1755, 927],
                            [-275, 225, 450, 450, 225, 45],
                            [80, 0 , 0 , 0 , 0 , 0]])/4320  
            elif knot_points[8] == knot_points[n-4]:
                M = np.array([[-45, 225, -450, 450, -225, 45],
                              [189, -765, 810, 630, -1755, 927],
                              [-360, 1080, 0, -2160, 0, 2376],
                              [360, -720, -720, 720, 1800, 936],
                              [-189, 180, 360, 360, 180, 36],
                              [45, 0 , 0 , 0 , 0 , 0]])/4320 
            elif knot_points[8] < knot_points[n-4]:
                M = np.array([[-45, 225, -450, 450, -225, 45],
                              [189, -765, 810, 630, -1755, 927],
                              [-360, 1080, 0, -2160, 0, 2376],
                              [360, -720, -720, 720, 1800, 936],
                              [-180, 180, 360, 360, 180, 36],
                              [36, 0 , 0 , 0 , 0 , 0]])/4320 
        elif i_t == n-4 and knot_points[8] < knot_points[n-4]:
            M = np.array([[-36, 180, -360, 360, -180, 36],
                          [180, -720, 720, 720, -1800, 936],
                          [-360, 1080, 0, -2160, 0, 2376],
                          [360, -720, -720, 720, 1800, 936],
                          [-189, 180, 360, 360, 180, 36],
                          [45, 0 , 0 , 0 , 0 , 0]])/4320 
        elif i_t == n-3 and knot_points[8] < knot_points[n-3]:
            M = np.array([[-36, 180, -360, 360, -180, 36],
                          [180, -720, 720, 720, -1800, 936],
                          [-360, 1080, 0, -2160, 0, 2376],
                          [411, -765, -810, 630, 1755, 927],
                          [-275, 225, 450, 450, 225, 45],
                          [80, 0, 0, 0, 0, 0]])/4320  
        elif i_t == n-2 and knot_points[8] < knot_points[n-2]:
            M = np.array([[-36, 180, -360, 360, -180, 36],
                          [180, -720, 720, 720, -1800, 936],
                          [-489, 1290, 240, -2280, -420, 2148],
                          [805, -1150, -1400, 400, 2000, 1120],
                          [-730, 400, 800, 800, 400, 80],
                          [270, 0, 0, 0, 0, 0]])/4320
        elif i_t == n-1:
            M = np.array([[-36, 180, -360, 360, -180, 36],
                          [411, -1155, 510, 1290, -1545, 489],
                          [-2075, 2875, 2050, -2650, -1975, 1775],
                          [5750, -3250, -4900, -1700, 2350, 1750],
                          [-8370, 1350, 2700, 2700, 1350, 270],
                          [4320, 0 , 0 , 0 , 0, 0]])/4320
        else:
            M = __get_5_order_matrix()
    return M