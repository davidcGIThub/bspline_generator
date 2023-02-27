import numpy as np
import matplotlib.pyplot as plt
from bsplinegenerator.helper_functions import get_dimension, get_time_to_point_correlation


def plot_minvo_curves_from_spline_data(order, curve_data, control_points):
    dimension = get_dimension(curve_data)
    if dimension == 3:
        plot_3d_minvo_curves(order, curve_data, control_points)
    elif dimension == 2:
        plot_2d_minvo_curves(order, curve_data, control_points)
    elif dimension == 1:
        plot_1d_minvo_curves(order, curve_data, control_points)
    else:
        plot_multidimensional_minvo_curve(order, curve_data, control_points)
    
def plot_3d_minvo_curves(order, curve_data, control_points):
    figure_title = str(order) + " Order Minvo Curves"
    plt.figure(figure_title)
    ax = plt.axes(projection='3d')
    ax.set_box_aspect(aspect =(1,1,1))
    ax.plot(curve_data[0,:], curve_data[1,:],curve_data[2,:],label="Minvo Curves")
    ax.scatter(np.array([]),np.array([]),np.array([]))
    ax.plot(control_points[0,:], control_points[1,:],control_points[2,:], color='r')
    ax.scatter(control_points[0,:], control_points[1,:],control_points[2,:],label="Minvo Control Points", color='r')
    ax.set_xlabel('x')
    ax.set_ylabel('r')
    ax.set_zlabel('z')
    plt.title(figure_title)
    plt.legend()
    plt.show()

def plot_2d_minvo_curves(order, curve_data, control_points):
    figure_title = str(order) + " Order Minvo Curves"
    plt.figure(figure_title)
    plt.plot(curve_data[0,:], curve_data[1,:],label="Piecewise Minvo Curves")
    plt.scatter(np.array([]),np.array([]))
    plt.plot(control_points[0,:], control_points[1,:],color='r')
    plt.scatter(control_points[0,:], control_points[1,:],linewidths=2,label="Minvo Control Points",color='r')
    plt.xlabel('x')
    plt.ylabel('r')
    plt.title(figure_title)
    plt.legend()
    plt.show()

def plot_1d_minvo_curves(order, curve_data, control_points):
    figure_title = str(order) + " Order Minvo Curves"
    curve_x_axis_data = get_time_to_point_correlation(curve_data,0,1)
    control_point_x_axis_data = get_time_to_point_correlation(control_points,0,1)
    plt.plot(curve_x_axis_data, curve_data,label="Minvo Curve")
    plt.scatter(np.array([]),np.array([]))
    plt.scatter(control_point_x_axis_data,control_points,color='r')
    plt.plot(control_point_x_axis_data,control_points,label="Minvo Control Points",color='r')
    plt.ylabel('curve data')
    plt.title(figure_title)
    plt.legend()
    plt.show()

def plot_multidimensional_minvo_curve(order, curve_data, control_points):
        figure_title = str(order) + " Order Minvo Curves"
        dimension = get_dimension(curve_data)
        curve_x_axis_data = get_time_to_point_correlation(curve_data,0,1)
        control_point_x_axis_data = get_time_to_point_correlation(control_points,0,1)
        plt.figure(figure_title)
        if(dimension > 1):
            for i in range(dimension):
                curve_label = "Dimension " + str(i)
                plt.plot(curve_x_axis_data, curve_data[i,:],label=curve_label)
                plt.scatter(control_point_x_axis_data,control_points[i,:],label="Minvo Control Points", color='r')
        else:
            raise Exception("Curve is 1 dimensional")
        plt.ylabel('curve data')
        plt.title(figure_title)
        plt.legend()
        plt.show()