from calendar import c
import numpy as np
import matplotlib.pyplot as plt
from pyrsistent import s
import scipy as sp
from bsplinegenerator.helper_functions import count_number_of_control_points, get_dimension, \
    get_time_to_point_correlation, get_dimension

def plot_bspline(spline_data, knot_point_values, control_points):
    dimension = get_dimension(spline_data)
    if dimension == 3:
        plot_3d_bspline(spline_data,knot_point_values, control_points)
    elif dimension == 2:
        plot_2d_bspline(spline_data, knot_point_values, control_points)
    elif dimension == 1:
        plot_1d_bspline(spline_data, knot_point_values, control_points)
    else:
        plot_multidimensional_bspline(spline_data, knot_point_values, control_points)
    
def plot_3d_bspline(spline_data, knot_point_values, control_points):
    order = get_order(knot_point_values, control_points)
    figure_title = str(order) + " Order B-Spline"
    plt.figure(figure_title)
    ax = plt.axes(projection='3d')
    ax.set_box_aspect(aspect =(1,1,1))
    ax.plot(spline_data[0,:], spline_data[1,:],spline_data[2,:],label="B-Spline")
    ax.scatter(knot_point_values[0,:], knot_point_values[1,:],knot_point_values[2,:],label="Spline at Knot Points")
    ax.plot(control_points[0,:], control_points[1,:],control_points[2,:])
    ax.scatter(control_points[0,:], control_points[1,:],control_points[2,:],label="Control Points")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title(figure_title)
    plt.legend()
    plt.show()

def plot_2d_bspline(spline_data, knot_point_values, control_points):
    order = get_order(knot_point_values, control_points)
    figure_title = str(order) + " Order B-Spline"
    plt.figure(figure_title)
    plt.plot(spline_data[0,:], spline_data[1,:],label="B-Spline")
    plt.scatter(knot_point_values[0,:], knot_point_values[1,:],label="Spline at Knot Points")
    plt.plot(control_points[0,:], control_points[1,:])
    plt.scatter(control_points[0,:], control_points[1,:],linewidths=2,label="Control Points")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(figure_title)
    plt.legend()
    plt.show()

def plot_1d_bspline(spline_data, knot_point_values, control_points):
    order = get_order(knot_point_values, control_points)
    figure_title = str(order) + " Order B-Spline"
    spline_x_axis_data = get_time_to_point_correlation(spline_data,0,1)
    control_point_x_axis_data = get_time_to_point_correlation(control_points,0,1)
    knot_point_x_axis_data = get_time_to_point_correlation(knot_point_values,0,1)
    plt.plot(spline_x_axis_data, spline_data,label="B-spline")
    plt.scatter(knot_point_x_axis_data, knot_point_values,label="Spline at Knot Points")
    plt.scatter(control_point_x_axis_data,control_points)
    plt.plot(control_point_x_axis_data,control_points,label="Control Points")
    plt.ylabel('spline data')
    plt.title(figure_title)
    plt.legend()
    plt.show()

def plot_multidimensional_bspline(spline_data, knot_point_values, control_points):
        order = get_order(knot_point_values, control_points)
        figure_title = str(order) + " Order B-Spline"
        dimension = get_dimension(spline_data)
        spline_x_axis_data = get_time_to_point_correlation(spline_data,0,1)
        control_point_x_axis_data = get_time_to_point_correlation(control_points,0,1)
        plt.figure(figure_title)
        if(dimension > 1):
            for i in range(dimension):
                spline_label = "Dimension " + str(i)
                plt.plot(spline_x_axis_data, spline_data[i,:],label=spline_label)
                plt.scatter(control_point_x_axis_data,control_points[i,:],label="Control Points")
        else:
            raise Exception("Spline is 1 dimensional")
        plt.ylabel('spline data')
        plt.title(figure_title)
        plt.legend()
        plt.show()

def get_order(defined_knot_point_values, control_points):
    dimension = get_dimension(control_points)
    num_control_points = count_number_of_control_points(control_points)
    if dimension > 1:
        num_defined_knot_points = np.shape(defined_knot_point_values)[1]
    else:
        num_defined_knot_points = len(defined_knot_point_values)
    num_intervals = num_defined_knot_points - 1
    order = num_control_points - num_intervals
    return order

def plot_bspline_vs_time(spline_data,time_data,knot_point_values,knot_points):
        figure_title = "B-Spline vs Time"
        dimension = get_dimension(spline_data)
        plt.figure(figure_title)
        if(dimension > 1):
            for i in range(dimension):
                spline_label = "Dimension " + str(i)
                plt.plot(time_data, spline_data[i,:],label=spline_label)
                plt.scatter(knot_points, knot_point_values[i,:])
        else:
            plt.plot(time_data, spline_data,label="Spline")
            plt.scatter(knot_points, knot_point_values)
        plt.xlabel('time')
        plt.ylabel('b(t)')
        plt.title(figure_title)
        plt.legend()
        plt.show()

def plot_bspline_basis_functions(basis_function_data, time_data, order):
    figure_title = "Basis Functions - Order " + str(order)
    plt.figure(figure_title)
    number_of_basis_functions = np.shape(basis_function_data)[0]
    for b in range(number_of_basis_functions):
        basis_label = "N" + str(b) + "," + str(order) + "(t)"
        basis_function = basis_function_data[b,:]
        plt.plot(time_data, basis_function, label=basis_label)
    plt.xlabel('time')
    plt.ylabel('N(t)')
    plt.title(figure_title)
    plt.legend(loc="center")
    plt.show()

def plot_bspline_derivative(derivative_order, derivative_data, derivative_at_knot_points, control_point_derivatives):
    dimension = get_dimension(derivative_data)
    if dimension == 3:
        plot_3d_bspline_derivative(derivative_order,derivative_data,derivative_at_knot_points,control_point_derivatives)
    elif dimension == 2:
        plot_2d_bspline_derivative(derivative_order,derivative_data,derivative_at_knot_points,control_point_derivatives)
    elif dimension == 1:
        plot_1d_bspline_derivative(derivative_order,derivative_data,derivative_at_knot_points,control_point_derivatives)
    else:
        plot_multidimensional_bspline_derivative(derivative_order,derivative_data,control_point_derivatives)

def plot_3d_bspline_derivative(derivative_order,derivative_data,derivative_at_knot_points,control_point_derivatives):
    figure_title = str(derivative_order) + " Order Derivative"
    plt.figure(figure_title)
    ax = plt.axes(projection='3d')
    ax.set_box_aspect(aspect =(1,1,1))
    ax.plot(derivative_data[0,:], derivative_data[1,:],derivative_data[2,:],label="B-Spline derivative")
    ax.scatter(derivative_at_knot_points[0,:], derivative_at_knot_points[1,:],derivative_at_knot_points[2,:],label="Derivative at Knot Points")
    ax.plot(control_point_derivatives[0,:],control_point_derivatives[1,:],control_point_derivatives[2,:])
    ax.scatter(control_point_derivatives[0,:],control_point_derivatives[1,:],control_point_derivatives[2,:],label="Control Point Derivatives")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title(figure_title)
    plt.legend()
    plt.show()

def plot_2d_bspline_derivative(derivative_order,derivative_data,derivative_at_knot_points,control_point_derivatives):
    figure_title = str(derivative_order) + " Order Derivative"
    plt.figure(figure_title)
    plt.plot(derivative_data[0,:], derivative_data[1,:],label="B-Spline derivative")
    plt.scatter(derivative_at_knot_points[0,:], derivative_at_knot_points[1,:],label="Derivative at Knot Points")
    plt.plot(control_point_derivatives[0,:], control_point_derivatives[1,:])
    plt.scatter(control_point_derivatives[0,:], control_point_derivatives[1,:],label="Control Point Derivatives")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(figure_title)
    plt.legend()
    plt.show()

def plot_1d_bspline_derivative(derivative_order,derivative_data,derivative_at_knot_points,control_point_derivatives):
    figure_title = str(derivative_order) + " Order Derivative"
    spline_x_axis_data = get_time_to_point_correlation(derivative_data,0,1)
    control_point_x_axis_data = get_time_to_point_correlation(control_point_derivatives,0,1)
    spline_at_knot_point_x_axis_data = get_time_to_point_correlation(derivative_at_knot_points,0,1)
    plt.plot(spline_x_axis_data, derivative_data,label="B-spline")
    plt.scatter(spline_at_knot_point_x_axis_data, derivative_at_knot_points,label="Derivative at Knot points")
    plt.plot(control_point_x_axis_data, control_point_derivatives)
    plt.scatter(control_point_x_axis_data, control_point_derivatives,label="Control Point Derivatives")
    plt.ylabel('spline derivative data')
    plt.title(figure_title)
    plt.legend()
    plt.show()

def plot_multidimensional_bspline_derivative(derivative_order,derivative_data,control_point_derivatives):
        figure_title = str(derivative_order) + " Order Derivative"
        dimension = get_dimension(derivative_data)
        spline_x_axis_data = get_time_to_point_correlation(derivative_data,0,1)
        control_point_x_axis_data = get_time_to_point_correlation(control_point_derivatives,0,1)
        plt.figure(figure_title)
        if(dimension > 1):
            for i in range(dimension):
                spline_label = "Dimension " + str(i)
                plt.plot(spline_x_axis_data, derivative_data[i,:],label=spline_label)
                plt.scatter(control_point_x_axis_data, control_point_derivatives[i,:])
        else:
            raise Exception("Spline is 1 dimensional")
        plt.ylabel('spline derivative data')
        plt.title(figure_title)
        plt.legend()
        plt.show()

def plot_bspline_derivative_vs_time(derivative_data, time_data, derivative_order):
    figure_title = str(derivative_order) + " Order Derivative vs Time"
    dimension = get_dimension(derivative_data)
    plt.figure(figure_title)
    if dimension > 1:
        for i in range(dimension):
            spline_label = "Dimension " + str(i)
            plt.plot(time_data, derivative_data[i,:],label=spline_label)
    else:
        plt.plot(time_data, derivative_data, label="Spline Derivative")
    plt.xlabel('time')
    plt.ylabel(str(derivative_order) + ' derivative')
    plt.title(figure_title)
    plt.legend()
    plt.show()

def plot_bspline_derivative_magnitude(magnitude_data, time_data, control_point_data, control_point_time):
    plt.figure("Derivative Magnitude")
    plt.plot(control_point_time, control_point_data, label="control point derivative norm", color = "tab:orange")
    plt.scatter(control_point_time, control_point_data, color = "tab:orange")
    plt.plot(time_data, magnitude_data, label = "derivative norm", color = "tab:blue")
    plt.xlabel('time')
    plt.ylabel('derivative magnitude')
    plt.title("Derivative Magnitude")
    plt.legend()
    plt.show()

def plot_bspline_curvature(curvature_data, time_data):
    plt.figure("Curvature")
    plt.plot(time_data, curvature_data, 'g')
    plt.scatter(np.array([]), np.array([]))
    plt.xlabel('time')
    plt.ylabel('curvature')
    plt.title("Curvature")
    plt.show()

def plot_bspline_centripetal_acceleration(centripetal_acceleration_data, time_data):
    plt.figure("Centripetal Acceleration")
    plt.plot(time_data, centripetal_acceleration_data, 'g')
    plt.scatter(np.array([]), np.array([]))
    plt.xlabel('time')
    plt.ylabel('centripetal acceleration')
    plt.title("Centripetal Acceleration")
    plt.show()

def plot_bspline_angular_rate(angular_rate_data, time_data):
    plt.figure("Angular Rate")
    plt.plot(time_data, angular_rate_data, 'g')
    plt.scatter(np.array([]), np.array([]))
    plt.xlabel('time')
    plt.ylabel('angular rate')
    plt.title("Angular Rate")
    plt.show()