import numpy as np

def get_dimension(control_points):
    if control_points.ndim == 1:
        dimension = 1
    else:
        dimension = len(control_points)
    return dimension

def count_number_of_control_points(control_points):
    if control_points.ndim == 1:
        number_of_control_points = len(control_points)
    else:
        number_of_control_points = len(control_points[0])
    return number_of_control_points

def calculate_number_of_control_points(order, knot_points):
    number_of_control_points = len(knot_points) - order - 1
    return number_of_control_points

def find_preceding_knot_index(time, order, knot_points):
        """ 
        This function finds the knot point preceding
        the current time
        """
        preceding_knot_index = -1
        number_of_control_points = calculate_number_of_control_points(order,knot_points)
        if time >= knot_points[number_of_control_points-1]:
            preceding_knot_index = number_of_control_points-1
        else:
            for knot_index in range(order,number_of_control_points+1):
                preceding_knot_index = number_of_control_points - 1
                knot_point = knot_points[knot_index]
                next_knot_point = knot_points[knot_index + 1]
                if time >= knot_point and time < next_knot_point:
                    preceding_knot_index = knot_index
                    break
        return preceding_knot_index

def find_end_time(control_points, knot_points):
    end_time = knot_points[count_number_of_control_points(control_points)]
    return end_time

def get_time_to_point_correlation(points,start_time,end_time):
    '''
    This is not a true correlation but distributes the points
    evenly through the time interval and provides a time to each point
    '''
    number_of_points = count_number_of_control_points(points)
    time_array = np.linspace(start_time, end_time, number_of_points)
    return time_array


def create_random_control_points_greater_than_angles(num_control_points,order,length,dimension):
    if order == 1:
        angle = np.pi/2
    elif order == 2:
        angle = np.pi/2
    elif order == 3:
        angle = np.pi/4
    elif order == 4:
        angle = np.pi/6
    elif order == 5:
        angle = np.pi/8
    control_points = np.zeros((dimension, num_control_points))
    for i in range(num_control_points):
        if i == 0:
            control_points[:,i][:,None] = np.array([[0],[0]])
        elif i == 1:
            random_vec = np.random.rand(2,1)
            next_vec = length*random_vec/np.linalg.norm(random_vec)
            control_points[:,i][:,None] = control_points[:,i-1][:,None] + next_vec
        else:
            new_angle = angle#*2*(0.5-np.random.rand())
            R = np.array([[np.cos(new_angle), -np.sin(new_angle)],[np.sin(new_angle), np.cos(new_angle)]])
            prev_vec = control_points[:,i-1][:,None] - control_points[:,i-2][:,None]
            unit_prev_vec = prev_vec/np.linalg.norm(prev_vec)
            next_vec = length*np.dot(R,unit_prev_vec)#*np.random.rand()
            control_points[:,i][:,None] = control_points[:,i-1][:,None] + next_vec
    return control_points