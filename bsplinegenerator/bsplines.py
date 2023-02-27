"""
This module contains code to evaluate an uniform b splines 
using the matrix method and the cox-de-boor table method for splines of order 
higher than the 5th degree. This also evaluates the derivatives of the B-spline
"""

import numpy as np
import sys
from bsplinegenerator.matrix_evaluation import matrix_bspline_evaluation, derivative_matrix_bspline_evaluation, \
    matrix_bspline_evaluation_for_dataset, matrix_bspline_derivative_evaluation_for_dataset
from bsplinegenerator.table_evaluation import table_bspline_evaluation, derivative_table_bspline_evaluation, \
    cox_de_boor_table_basis_function
from bsplinegenerator.helper_functions import count_number_of_control_points, get_dimension, get_time_to_point_correlation
from bsplinegenerator.bspline_plotter import plot_bspline, plot_bspline_vs_time, plot_bspline_basis_functions, \
    plot_bspline_derivative_vs_time, plot_bspline_derivative_magnitude, plot_bspline_curvature, plot_bspline_angular_rate, \
    plot_bspline_centripetal_acceleration, plot_bspline_derivative
from bsplinegenerator.bspline_to_bezier import convert_list_to_bezier_control_points
from bsplinegenerator.bezier_curve_plotter import plot_bezier_curves_from_spline_data
from bsplinegenerator.bspline_to_minvo import convert_list_to_minvo_control_points
from bsplinegenerator.minvo_curve_plotter import plot_minvo_curves_from_spline_data

class BsplineEvaluation:
    """
    This class contains contains code to evaluate uniform b spline 
    using the matrix method and the cox-de-boor table method for splines of order
    higher than the 5th degree. This also uses the table method for B-splines
    of order higher than 5. This also evaluates the derivatives of the B-spline.
    """

    def __init__(self, control_points, order, start_time, scale_factor=1, clamped=False):
        '''
        Constructor for the BsplinEvaluation class, each column of
        control_points is a control point. Start time should be an integer.
        '''
        self._control_points = control_points
        self._num_control_points = count_number_of_control_points(self._control_points)
        if order >= self._num_control_points:
            raise Exception("Polynomial of order " , order, " needs at least " , order + 1 , " control points")
        self._order = order
        self._scale_factor = scale_factor
        self._start_time = start_time
        self._clamped = clamped
        if clamped:
            self._knot_points = self.__create_clamped_knot_points()
        else:
            self._knot_points = self.__create_knot_points()
        self._end_time = self._knot_points[self._num_control_points]

    def get_start_time(self):
        return self._start_time

    def get_end_time(self):
        return self._end_time

    def get_spline_data(self , num_data_points_per_interval):
        '''
        Returns equally distributed data points for the spline, as well
        as time data for the parameterization
        '''
        number_of_data_points = num_data_points_per_interval*(self._num_control_points-self._order) + 1
        time_data = np.linspace(self._start_time, self._end_time, number_of_data_points)
        if self._order > 7 or (self._clamped and self._order > 5):
            spline_data = self.__get_spline_data_point_by_point_method(time_data)
        else:
            spline_data = matrix_bspline_evaluation_for_dataset(self._control_points, self._knot_points, num_data_points_per_interval, self._clamped)
        return spline_data, time_data

    def __get_spline_data_point_by_point_method(self,time_data):
        number_of_data_points = len(time_data)
        dimension = get_dimension(self._control_points)
        if dimension == 1:
            spline_data = np.zeros(number_of_data_points)
        else:
            spline_data = np.zeros((dimension,number_of_data_points))
        for i in range(number_of_data_points):
            t = time_data[i]
            if dimension == 1:
                spline_data[i] = self.get_spline_at_time_t(t)
            else:
                spline_data[:,i][:,None] = self.get_spline_at_time_t(t)
        return spline_data

    def get_spline_derivative_data(self,num_data_points_per_interval, rth_derivative):
        '''
        Returns equally distributed data points for the derivative of the spline, 
        as well as time data for the parameterization
        '''
        number_of_data_points = num_data_points_per_interval*(self._num_control_points-self._order) + 1
        time_data = np.linspace(self._start_time, self._end_time, number_of_data_points)
        if self._order > 5 or (self._clamped and self._order > 7):
            spline_derivative_data = self.__get_spline_derivative_data_point_by_point_method(rth_derivative,time_data)
        else:
            spline_derivative_data = matrix_bspline_derivative_evaluation_for_dataset(rth_derivative, self._scale_factor, self._control_points, self._knot_points, num_data_points_per_interval, self._clamped)
        return spline_derivative_data, time_data

    def __get_spline_derivative_data_point_by_point_method(self,rth_derivative,time_data):
        number_of_data_points = len(time_data)
        time_data = np.linspace(self._start_time, self._end_time, number_of_data_points)
        dimension = get_dimension(self._control_points)
        if dimension == 1:
            spline_derivative_data = np.zeros(number_of_data_points)
        else:
            spline_derivative_data = np.zeros((dimension,number_of_data_points))
        for i in range(number_of_data_points):
            t = time_data[i]
            if dimension == 1:
                spline_derivative_data[i] = self.get_derivative_at_time_t(t, rth_derivative)
            else:
                spline_derivative_data[:,i][:,None] = self.get_derivative_at_time_t(t, rth_derivative)
        return spline_derivative_data

    def get_derivative_magnitude_data(self, num_data_points_per_interval, derivative_order):
        '''
        Returns equally distributed data points for the magnitude of the rth derivative
        '''
        derivative_data, time_data = self.get_spline_derivative_data(num_data_points_per_interval, derivative_order)
        dimension = get_dimension(self._control_points)
        if dimension > 1:
            derivative_magnitude_data = np.linalg.norm(derivative_data,2,0)
        else:
            derivative_magnitude_data = np.abs(derivative_data)
        return derivative_magnitude_data, time_data

    def get_spline_curvature_data(self, num_data_points_per_interval):
        '''
        Returns equally distributed data points for the curvature of the spline, 
        as well as time data for the parameterization
        '''
        angular_rate_data, time_data = self.get_angular_rate_data(num_data_points_per_interval)
        velocity_data, time_data = self.get_spline_derivative_data(num_data_points_per_interval,1)
        dimension = get_dimension(self._control_points)
        if dimension == 1:
            velocity_magnitude = np.linalg.norm(np.vstack((np.ones(len(time_data)), velocity_data)),2,0)
        else:
            velocity_magnitude = np.linalg.norm(velocity_data,2,0)
        undefined_indices = velocity_magnitude < 1e-10
        velocity_magnitude[undefined_indices] = 1
        curvature_data = angular_rate_data/velocity_magnitude
        return curvature_data, time_data

    def get_angular_rate_data(self,num_data_points_per_interval):
        '''
        Returns equally distributed data points for the angular rate
        of the spline, as well as time data for the parameterization
        '''
        number_of_data_points = num_data_points_per_interval*(self._num_control_points-self._order) + 1
        centripetal_acceleration_data, time_data = self.get_centripetal_acceleration_data(num_data_points_per_interval)
        velocity_data, time_data = self.get_spline_derivative_data(num_data_points_per_interval,1)
        dimension = get_dimension(self._control_points)
        if dimension == 1:
            velocity_magnitude = np.linalg.norm(np.vstack((np.ones(len(time_data)), velocity_data)),2,0)
        else:
            velocity_magnitude = np.linalg.norm(velocity_data,2,0)
        undefined_indices = velocity_magnitude < 1e-10
        velocity_magnitude_without_zeros = np.copy(velocity_magnitude)
        velocity_magnitude_without_zeros[undefined_indices] = 1
        angular_rate_data = centripetal_acceleration_data/velocity_magnitude_without_zeros
        angular_rate_data[undefined_indices] = 0
        # this section of the code checks for colinear 180 degree turns
        if any(velocity_magnitude < 1):
            bspline_data, time_data = self.get_spline_data(num_data_points_per_interval)
            if dimension == 1:
                bspline_data = np.vstack((np.linspace(self._start_time, self._end_time,number_of_data_points), bspline_data))
            prev_spline_data = bspline_data[:,0:-1]
            next_spline_data = bspline_data[:,1:]
            spline_vectors = next_spline_data-prev_spline_data
            norm_spline_vectors = np.linalg.norm(spline_vectors,2,0)
            colinear_instances = np.abs(spline_vectors[:,1:])*norm_spline_vectors[0:-1] == \
                np.abs(spline_vectors[:,0:-1])*norm_spline_vectors[1:]
            colinear_ascending_instances = spline_vectors[:,1:]*norm_spline_vectors[0:-1] == \
                spline_vectors[:,0:-1]*norm_spline_vectors[1:]
            colinear_unordered = np.any(colinear_instances != colinear_ascending_instances,0)
            indices_180_degree_turns = np.where(colinear_unordered==True)[0] + 1
            angular_rate_data[indices_180_degree_turns] = sys.maxsize
        if self._order == 1:
            num_intervals = self._num_control_points - self._order
            knot_point_indices = np.arange(1,num_intervals)*num_data_points_per_interval
            angular_rate_data[knot_point_indices] = sys.maxsize
        return angular_rate_data, time_data

    def get_centripetal_acceleration_data(self,num_data_points_per_interval):
        '''
        Returns equally distributed data points for the centripetal acceleration
        of the spline, as well as time data for the parameterization
        '''
        dimension = get_dimension(self._control_points)
        if dimension > 3:
            raise Exception("Centripetal acceleration cannot be evaluated for higher than 3 dimensions")
        else:
            velocity_data, time_data = self.get_spline_derivative_data(num_data_points_per_interval,1)
            acceleration_data, time_data = self.get_spline_derivative_data(num_data_points_per_interval,2)
            if dimension == 1:
                velocity_matrix = np.vstack((np.ones(len(time_data)), velocity_data)).T
                acceleration_matrix = np.vstack((np.zeros(len(time_data)), acceleration_data)).T
                velocity_magnitude = np.linalg.norm(velocity_matrix.T,2,0)
                cross_product_norm = np.abs(np.cross(velocity_matrix, acceleration_matrix).flatten())
            elif dimension == 2:
                velocity_matrix = velocity_data.T
                acceleration_matrix = acceleration_data.T
                velocity_magnitude = np.linalg.norm(velocity_data,2,0)
                cross_product_norm = np.abs(np.cross(velocity_matrix, acceleration_matrix).flatten())
            else:
                velocity_matrix = velocity_data.T
                acceleration_matrix = acceleration_data.T
                velocity_magnitude = np.linalg.norm(velocity_data,2,0)
                cross_product_norm = np.linalg.norm(np.cross(velocity_matrix, acceleration_matrix),2,1).flatten()
            undefined_indices = np.where(velocity_magnitude < 1e-10)[0]
            velocity_magnitude[undefined_indices] = 1
            centripetal_acceleration = cross_product_norm/velocity_magnitude
            centripetal_acceleration[undefined_indices] = 0
            return centripetal_acceleration, time_data

    def get_basis_function_data(self, num_data_points_per_interval):
        ''' 
        Returns arrays of (num_basis_functions x num_data_points) of the basis
        functions.
        '''
        number_of_data_points = num_data_points_per_interval*(self._num_control_points-self._order) + 1
        num_basis_functions = self._num_control_points
        time_data = np.linspace(self._start_time, self._end_time, number_of_data_points)
        basis_function_data = np.zeros((num_basis_functions, number_of_data_points))
        for j in range(number_of_data_points):
            t = time_data[j]
            basis_function_data[:,j][:,None] = self.get_basis_functions_at_time_t(t)
        return basis_function_data, time_data

    def get_spline_at_time_t(self, time):
        """
        This function evaluates the B spline at the given time
        """
        if (self._clamped and self._order > 5) or self._order > 7:
            spline_at_time_t = table_bspline_evaluation(time, self._control_points, self._knot_points, self._clamped)
        else:
            spline_at_time_t = matrix_bspline_evaluation(time, self._scale_factor, self._control_points, self._knot_points, self._clamped)
        return spline_at_time_t

    def get_derivative_at_time_t(self, time, derivative_order):
        '''
        This function evaluates the rth derivative of the spline at time t
        '''
        if (self._clamped and self._order > 5) or self._order > 7:
            derivative_at_time_t = derivative_table_bspline_evaluation(time, derivative_order, self._control_points, self._knot_points, self._clamped)       
        else:
            derivative_at_time_t = derivative_matrix_bspline_evaluation(time, derivative_order, self._scale_factor, self._control_points, self._knot_points, self._clamped)
        return derivative_at_time_t

    def get_derivative_magnitude_at_time_t(self, time, derivative_order):
        '''
        This function evaluates the rth derivative magnitude of the spline at time t
        '''
        derivative = self.get_derivative_at_time_t(time,derivative_order)
        magnitude = np.linalg.norm(derivative)
        return magnitude

    def get_control_point_derivatives(self, derivative_order):
        dimension = get_dimension(self._control_points)
        if derivative_order > self._order:
            raise Exception("Derivative order higher than degree of spline")
        elif self._clamped:
            points = self.__get_clamped_control_point_derivatives(derivative_order, dimension)
        else:
            points = self.__get_open_control_point_derivatives(derivative_order, dimension)
        return points

    def __get_open_control_point_derivatives(self, derivative_order, dimension):
        points = self._control_points
        num_control_points = self._num_control_points
        for i in range(derivative_order):
            if dimension > 1:
                current = points[:,0:num_control_points-1-i]
                next = points[:,1:num_control_points-i]
            else:
                current = points[0:num_control_points-1-i]
                next = points[1:num_control_points-i]
            points = (next - current)/self._scale_factor
        return points

    def __get_clamped_control_point_derivatives(self, derivative_order, dimension):
        points = self._control_points
        num_control_points = self._num_control_points
        num_intervals = num_control_points - self._order
        for r in range(1,derivative_order+1):
            if dimension > 1:
                new_points = np.zeros((dimension, num_control_points - r))
                for i in range(num_control_points - r):
                        point_1 = points[:,i]
                        point_2 = points[:,i+1]
                        new_points[:,i] = (self._order-r+1) * (point_2 - point_1) / \
                            (self._scale_factor * np.min((num_intervals,self._order-r+1, i+1, num_control_points-r-i)))
            else:
                new_points = np.zeros(num_control_points - r)
                for i in range(num_control_points - r):
                        point_1 = points[i]
                        point_2 = points[i+1]
                        new_points[i] = (self._order-r+1) * (point_2 - point_1) / \
                            (self._scale_factor * np.min((num_intervals,self._order-r+1, i+1, num_control_points-r-i)))
            points = new_points
        return points

    def get_control_point_derivative_magnitude_data(self, derivative_order):
        control_point_derivative_data = self.get_control_point_derivatives(derivative_order)
        dimension = get_dimension(self._control_points)
        if dimension > 1:
            point_magnitude_data = np.linalg.norm(control_point_derivative_data,2,0)
        else:
            point_magnitude_data = np.abs(control_point_derivative_data)
        control_point_time_data = get_time_to_point_correlation(point_magnitude_data,self._start_time,self._end_time)
        return point_magnitude_data, control_point_time_data

    def get_curvature_at_time_t(self, time):
        '''
        This function evaluates the curvature at time t
        '''
        dimension = get_dimension(self._control_points)
        velocity = self.get_derivative_at_time_t(time,1)
        if dimension > 3:
            raise Exception("Curvature cannot be evaluated for higher than 3 dimensions")
        if dimension == 1:
            velocity_magnitude = np.linalg.norm(np.array([1 , velocity]))
        else:
            velocity_magnitude = np.linalg.norm(velocity)
        if velocity_magnitude < 1e-10:
            velocity_magnitude == 1
        angular_rate = self.get_angular_rate_at_time_t(time)
        curvature = angular_rate/velocity_magnitude
        return curvature

    def get_angular_rate_at_time_t(self, time):
        '''
        This function evaluates the angular rate at time t
        '''
        dimension = get_dimension(self._control_points)
        velocity = self.get_derivative_at_time_t(time,1)
        if dimension > 3:
            raise Exception("Curvature cannot be evaluated for higher than 3 dimensions")
        if dimension == 1:
            velocity_magnitude = np.linalg.norm(np.array([1 , velocity]))
        else:
            velocity_magnitude = np.linalg.norm(velocity)
        centripetal_acceleration = self.get_centripetal_acceleration_at_time_t(time)
        if velocity_magnitude < 0.1:
            num_intervals = self._num_control_points - self._order
            time_step = (self._end_time - self._start_time)/(100*num_intervals)
            if time == self._start_time:
                point_a = self.get_spline_at_time_t(time)
                point_b = self.get_spline_at_time_t(time+time_step)
                point_c = self.get_spline_at_time_t(time+2*time_step)
            elif time == self._end_time:
                point_a = self.get_spline_at_time_t(time-time_step*2)
                point_b = self.get_spline_at_time_t(time-time_step)
                point_c = self.get_spline_at_time_t(time)
            else:
                point_a = self.get_spline_at_time_t(time-time_step)
                point_b = self.get_spline_at_time_t(time)
                point_c = self.get_spline_at_time_t(time+time_step)
            result = self.check_if_points_are_ascending_colinear(point_a, point_b, point_c)
            if result == "colinear_ascending":
                angular_rate = 0
            elif result == "colinear_unordered":
                angular_rate = sys.maxsize
            elif velocity_magnitude < 1e-10:
                angular_rate = 0
            else:
                angular_rate = centripetal_acceleration/velocity_magnitude
        else:
            angular_rate = centripetal_acceleration/velocity_magnitude
        return angular_rate

    def get_centripetal_acceleration_at_time_t(self,time):
        '''
        This function evaluates the centripetal acceleration at time t
        '''
        dimension = get_dimension(self._control_points)
        if dimension > 3:
            raise Exception("Centripetal acceleration cannot be evaluated for higher than 3 dimensions")
        if dimension == 1:
            derivative_vector = np.array([1 , self.get_derivative_at_time_t(time,1)[0]])
            derivative_2nd_vector = np.array([0 , self.get_derivative_at_time_t(time,2)[0]])
        else:
            derivative_vector = self.get_derivative_at_time_t(time,1)
            derivative_2nd_vector = self.get_derivative_at_time_t(time,2)
        derivative_magnitude = np.linalg.norm(derivative_vector)
        if derivative_magnitude < 1e-10:
            return 0
        centripetal_acceleration = np.linalg.norm(np.cross(derivative_vector.flatten(), derivative_2nd_vector.flatten())) / derivative_magnitude
        return centripetal_acceleration

    def get_basis_functions_at_time_t(self,time):
        '''
        Returns the values for each basis function at time t
        '''
        end_time = self._end_time
        num_basis_functions = self._num_control_points
        basis_functions_at_time_t = np.zeros((num_basis_functions,  1))
        for i in range(num_basis_functions):
            basis_functions_at_time_t[i,0] = cox_de_boor_table_basis_function(time, i, self._order , self._knot_points, end_time, self._clamped)
        return basis_functions_at_time_t

    def get_defined_knot_points(self):
        '''
        returns the knot points that are defined along the curve
        '''
        number_of_control_points = self._num_control_points
        defined_knot_points = self._knot_points[self._order:number_of_control_points+1]
        return defined_knot_points

    def get_knot_points(self):
        '''
        returns all the knot points
        '''
        return self._knot_points

    def get_spline_at_knot_points(self):
        '''
        Returns spline data evaluated at the knot points for
        which the spline is defined.
        '''
        time_data = self.get_defined_knot_points()
        number_of_data_points = len(time_data)
        dimension = get_dimension(self._control_points)
        if dimension == 1:
            spline_data = np.zeros(number_of_data_points)
        else:
            spline_data = np.zeros((dimension,number_of_data_points))
        for i in range(number_of_data_points):
            t = time_data[i]
            if dimension == 1:
                spline_data[i] = self.get_spline_at_time_t(t)
            else:
                spline_data[:,i][:,None] = self.get_spline_at_time_t(t)
        return spline_data, time_data

    def get_spline_derivative_at_knot_points(self,derivative_order):
        '''
        Returns spline derivative data evaluated at the knot points for
        which the spline is defined.
        '''
        time_data = self.get_defined_knot_points()
        number_of_data_points = len(time_data)
        dimension = get_dimension(self._control_points)
        if dimension == 1:
            derivative_data = np.zeros(number_of_data_points)
        else:
            derivative_data = np.zeros((dimension,number_of_data_points))
        for i in range(number_of_data_points):
            t = time_data[i]
            if dimension == 1:
                derivative_data[i] = self.get_derivative_at_time_t(t,derivative_order)
            else:
                derivative_data[:,i][:,None] = self.get_derivative_at_time_t(t,derivative_order)
        return derivative_data, time_data

    def get_bezier_control_points(self):
        if self._order > 7:
            print("Package not capable of converting control points for spline of order higher than 5")
            return np.array([])
        elif self._clamped:
            if self._num_control_points == self._order + 1:
                return self._control_points
            else:
                print("Package not capable of converting clamped control points with more than one interval.")
                return np.array([])
        else:
            bezier_control_points = convert_list_to_bezier_control_points(self._control_points,self._order)
            return bezier_control_points
        
    def get_minvo_control_points(self):
        if self._order > 7:
            print("Package not capable of converting control points for spline of order higher than 7")
            return np.array([])
        elif self._clamped:
            if self._num_control_points == self._order + 1:
                return self._control_points
            else:
                print("Package not capable of converting clamped control points with more than one interval.")
                return np.array([])
        else:
            minvo_control_points = convert_list_to_minvo_control_points(self._control_points,self._order)
            return minvo_control_points

    def __create_knot_points(self):
        '''
        This function creates evenly distributed knot points
        '''
        number_of_control_points = self._num_control_points
        number_of_knot_points = number_of_control_points + self._order + 1
        knot_points = np.arange(number_of_knot_points)*self._scale_factor + self._start_time - self._order*self._scale_factor
        return knot_points

    def __create_clamped_knot_points(self):
        """ 
        Creates the list of knot points in the closed interval [t_0, t_{k+p}] 
        with the first k points equal to t_k and the last k points equal to t_{p}
        where k = order of the polynomial, and p = number of control points
        """
        number_of_control_points = self._num_control_points
        number_of_knot_points = number_of_control_points + self._order + 1
        number_of_unique_knot_points = number_of_knot_points - 2*self._order
        unique_knot_points = np.arange(0,number_of_unique_knot_points) * self._scale_factor + self._start_time
        knot_points = np.zeros(number_of_knot_points) + self._start_time
        knot_points[self._order : self._order + number_of_unique_knot_points] = unique_knot_points
        knot_points[self._order + number_of_unique_knot_points: 2*self._order + number_of_unique_knot_points] = unique_knot_points[-1]
        return knot_points

    def plot_spline(self, num_data_points_per_interval):
        spline_data, time_data = self.get_spline_data(num_data_points_per_interval)
        spline_at_knot_points, defined_knot_points = self.get_spline_at_knot_points()
        plot_bspline(spline_data, spline_at_knot_points, self._control_points)

    def plot_spline_vs_time(self, num_data_points_per_interval):
        spline_data, time_data = self.get_spline_data(num_data_points_per_interval)
        spline_at_knot_points, defined_knot_points = self.get_spline_at_knot_points()
        plot_bspline_vs_time(spline_data,time_data,spline_at_knot_points,defined_knot_points)
       
    def plot_basis_functions(self, num_data_points_per_interval):
        basis_function_data, time_data = self.get_basis_function_data(num_data_points_per_interval)
        plot_bspline_basis_functions(basis_function_data, time_data, self._order)

    def plot_derivative(self, num_data_points_per_interval, derivative_order):
        spline_derivative_data, time_data = self.get_spline_derivative_data(num_data_points_per_interval,derivative_order)
        control_point_derivatives = self.get_control_point_derivatives(derivative_order)
        derivative_at_knot_points, defined_knot_points = self.get_spline_derivative_at_knot_points(derivative_order)
        plot_bspline_derivative(derivative_order, spline_derivative_data, derivative_at_knot_points, control_point_derivatives)

    def plot_derivative_vs_time(self, num_data_points_per_interval, derivative_order):
        spline_derivative_data, time_data = self.get_spline_derivative_data(num_data_points_per_interval,derivative_order)
        plot_bspline_derivative_vs_time(spline_derivative_data, time_data, derivative_order)

    def plot_derivative_magnitude(self, num_data_points_per_interval, derivative_order):
        derivative_magnitude_data, time_data = self.get_derivative_magnitude_data(num_data_points_per_interval,derivative_order)
        control_point_derivative_magnitude_data, control_point_time_data = self.get_control_point_derivative_magnitude_data(derivative_order)
        plot_bspline_derivative_magnitude(derivative_magnitude_data, time_data, control_point_derivative_magnitude_data, control_point_time_data)

    def plot_curvature(self, num_data_points_per_interval):
        spline_curvature_data, time_data = self.get_spline_curvature_data(num_data_points_per_interval)
        plot_bspline_curvature(spline_curvature_data, time_data)

    def plot_angular_rate(self, num_data_points_per_interval):
        angular_rate_data, time_data = self.get_angular_rate_data(num_data_points_per_interval)
        plot_bspline_angular_rate(angular_rate_data, time_data)

    def plot_centripetal_acceleration(self, num_data_points_per_interval):
        centripetal_acceleration_data, time_data = self.get_centripetal_acceleration_data(num_data_points_per_interval)
        plot_bspline_centripetal_acceleration(centripetal_acceleration_data, time_data)

    def plot_bezier_curves(self, num_data_points_per_interval):
        if self._order > 7:
            print("Package not capable of converting control points for spline of order higher than 7")
        elif self._clamped and self._num_control_points != self._order + 1:
            print("Package not capable of converting clamped control points with more than one interval.")
        else:
            bezier_control_points = self.get_bezier_control_points()
            spline_data, time_data = self.get_spline_data(num_data_points_per_interval)
            plot_bezier_curves_from_spline_data(self._order, spline_data, bezier_control_points)

    def plot_minvo_curves(self, num_data_points_per_interval):
        if self._clamped and self._num_control_points != self._order + 1:
            print("Package not capable of converting clamped control points with more than one interval.")
        if self._order > 7:
            print("Package not capable of converting control points for spline of order higher than 7")
        else:
            minvo_control_points = self.get_minvo_control_points()
            spline_data, time_data = self.get_spline_data(num_data_points_per_interval)
            plot_minvo_curves_from_spline_data(self._order, spline_data, minvo_control_points)

    def check_if_points_are_ascending_colinear(self, point_a, point_b, point_c):
        vector_1 = point_b-point_a
        vector_2 = point_c-point_b
        norm_1 = np.linalg.norm(vector_1)
        norm_2 = np.linalg.norm(vector_2)
        term_1 = norm_1*vector_2
        term_2 = norm_2*vector_1
        if  np.array_equal(term_1, term_2):
            result = "colinear_ascending"
        elif np.array_equal(np.abs(term_1), np.abs(term_2)):
            result = "colinear_unordered"
        else:
            result = "not_colinear"
        return result