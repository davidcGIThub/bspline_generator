
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